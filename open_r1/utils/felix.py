import re
from datasets import Dataset, load_dataset, DatasetDict
from typing import Optional, Union, Dict, List
import json
from pathlib import Path

def load_dataset_from_source(
    source: Union[str, Path],
    data_format: str = "jsonl",
    split_ratio: float = 0.05,
    seed: int = 42,
    name: Optional[str] = None,
    **kwargs
) -> Dict[str, Dataset]:
    """
    加载数据集，支持本地jsonl文件或Huggingface数据集。

    Args:
        source: 数据源路径或Huggingface数据集名称
        data_format: 数据格式，支持"jsonl"或"hf"
        split_ratio: 测试集比例，默认0.1
        seed: 随机种子
        name: HuggingFace数据集配置名称
        **kwargs: 其他参数传递给load_dataset

    Returns:
        DatasetDict: 包含'train'和'test'分割的数据集
    """
    if data_format == "jsonl":
        # 加载本地jsonl文件
        source = Path(source)
        if not source.exists():
            raise FileNotFoundError(f"找不到文件：{source}")
        
        # 读取jsonl文件
        data = []
        with open(source, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    data.append(item)
                except json.JSONDecodeError as e:
                    print(f"警告：跳过无效的JSON行：{line.strip()}\n错误：{e}")
        
        # 创建Dataset对象
        dataset = Dataset.from_list(data)
        
        # 分割数据集
        splits = dataset.train_test_split(
            test_size=split_ratio,
            seed=seed,
            shuffle=True
        )
        
        # 返回DatasetDict而不是普通字典
        return DatasetDict({
            "train": splits["train"],
            "test": splits["test"]
        })
    
    elif data_format == "hf":
        # 加载HuggingFace数据集
        dataset = load_dataset(source, name=name, **kwargs)
        return dataset
    
    else:
        raise ValueError(f"不支持的数据格式：{data_format}")


SYSTEM_PROMPT = (
    "作为由泛联新安开发的具备深度思考能力的智能编程助手DTCoder。您的任务是用户提出问题，DTCoder解决问题。DTCoder首先在心中思考推理过程，然后向用户提供答案。"
    "推理过程采用<think> </think>标签包围，"
    "<think>这里是推理过程</think>这里是答案"
)


def make_conversation(example):
        system_content = example.get("system", SYSTEM_PROMPT)
        return {
            "prompt": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": example["problem"]},
            ],
        }


def format_reward(prompts, completions, **kwargs):
    """Reward function that checks if the completion has a specific format based on system prompt."""
    pattern = r"^<think>.*?</think>\s*.*$"
    completion_contents = [completion[0]["content"] for completion in completions]
    
    # 检查每个 prompt 中的 system content
    system_contents = [
        prompt[0]["content"] if isinstance(prompt, list) and prompt 
        and prompt[0]["role"] == "system" else ""
        for prompt in prompts
    ]
    
    rewards = []
    for system_content, content in zip(system_contents, completion_contents):
        # 检查是否匹配 think-answer 格式
        match = bool(re.match(pattern, content, re.DOTALL | re.MULTILINE))
        
        # 检查系统提示中是否包含相关关键字
        requires_think = any(keyword in system_content for keyword 
                           in ["深度思考", "<think>", "思考推理"])
        
        if requires_think:
            # 如果系统提示要求思考，那么需要匹配格式
            reward = 1.0 if match else 0.0
        else:
            # 如果系统提示没有要求思考，那么不应该使用 think 格式
            reward = 1.0 if not match else 0.0
            
        rewards.append(reward)
    
    return rewards

def reasoning_steps_reward(completions, **kwargs):
    r"""Reward function that checks for clear step-by-step reasoning.
    Regex pattern:
        英文模式:
            Step \d+: - matches "Step 1:", "Step 2:", etc.
            ^\d+\. - matches numbered lists like "1.", "2.", etc. at start of line
            \n- - matches bullet points with hyphens
            \n\* - matches bullet points with asterisks
            First,|Second,|Next,|Finally, - matches transition words
        中文模式:
            第\d+步 - 匹配 "第1步"，"第2步" 等
            步骤\d+ - 匹配 "步骤1"，"步骤2" 等
            首先|然后|接着|最后 - 匹配中文过渡词
    """
    pattern = r"(Step\s*\d+[：:]?|^\d+\.|\n[-*]|(?:First|Second|Next|Finally)[,，]|第\d+步[：:]?|步骤\s*\d+[：:]?|(?:首先|然后|接着|最后)[，,])"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [len(re.findall(pattern, content)) for content in completion_contents]

    # Magic nubmer 3 to encourage 3 steps and more, otherwise partial reward
    return [min(1.0, count / 3) for count in matches]