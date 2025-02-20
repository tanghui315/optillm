from typing import Optional, Union, Any
import openai
import torch
from trl import GRPOTrainer
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from datasets import Dataset, IterableDataset
from trl.models import  unwrap_model_for_generation
from trl.data_utils import maybe_apply_chat_template, is_conversational,apply_chat_template
from open_r1.configs import GRPOConfig
from trl.trainer.grpo_trainer import RewardFunc

class RemoteAPIGRPOTrainer(GRPOTrainer):
    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        api_base: str = "http://36.103.203.24:5903/v1",
        api_key: str = "sk-xxx",
        api_model: str = "dtcoder",
        max_retries: int = 3,
        use_remote_vllm: bool = False,
        args: GRPOConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        **kwargs
    ):
        # 初始化父类
        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_classes=reward_processing_classes,
            **kwargs
        )
        
        # 远程API配置
        self.api_config = {
            "base_url": api_base,
            "api_key": api_key,
            "model": api_model,
            "max_retries": max_retries
        }
        self.use_remote_vllm = use_remote_vllm
        
        # 初始化OpenAI客户端
        self.client = openai.Client(
            base_url=self.api_config["base_url"],
            api_key=self.api_config["api_key"]
        )
        
    def _call_remote_api(self, inputs):
        """调用远程API的核心方法
        
        Args:
            inputs: 包含原始messages的输入数据
        """
        responses = []
        for input_item in inputs:
            # 从输入中提取messages
            messages = []
            
            # 1. 添加系统提示(如果存在)
            if "system" in input_item:
                messages.append({
                    "role": "system",
                    "content": input_item["system"]
                })
            
            # 2. 添加用户消息
            if isinstance(input_item["prompt"], list):
                # 如果prompt是消息列表，直接使用
                messages.extend(input_item["prompt"])
            else:
                # 如果是单个文本，构造用户消息
                messages.append({
                    "role": "user",
                    "content": input_item["prompt"]
                })

            for _ in range(self.api_config["max_retries"]):
                try:
                    response = self.client.chat.completions.create(
                        model=self.api_config["model"],
                        messages=messages,
                        temperature=self.args.temperature,
                        max_tokens=self.max_completion_length,
                        n=self.num_generations
                    )
                    completions = [choice.message.content for choice in response.choices]
                    responses.extend(completions)
                    break
                except Exception as e:
                    if _ == self.api_config["max_retries"] - 1:
                        self.accelerator.print(f"API调用失败: {str(e)}")
                        responses.extend([""] * self.num_generations)
        return responses

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The RemoteAPIGRPOTrainer does not support returning outputs")

        device = self.accelerator.device
        prompts = [x["prompt"] for x in inputs]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        prompt_inputs = self.processing_class(
            prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)

        if self.max_prompt_length is not None:
            prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -self.max_prompt_length:]
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -self.max_prompt_length:]

        # 根据配置选择生成方式
        if self.use_vllm:
            # 使用原始vLLM生成
            return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
        elif self.use_remote_vllm:
            # 使用远程API生成
            # 保留原始输入数据，包含system prompt
            completions = self._call_remote_api(inputs)
            
            # 将生成的文本转换为token IDs
            completion_inputs = self.processing_class(
                completions, 
                return_tensors="pt",
                padding=True,
                add_special_tokens=False
            ).to(device)
            
            completion_ids = completion_inputs["input_ids"]
            prompt_inputs_repeated = torch.repeat_interleave(prompt_inputs["input_ids"], self.num_generations, dim=0)
            prompt_completion_ids = torch.cat([prompt_inputs_repeated, completion_ids], dim=1)
        else:
            # 使用常规生成
            with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                prompt_completion_ids = unwrapped_model.generate(
                    **prompt_inputs, generation_config=self.generation_config
                )

        prompt_length = prompt_inputs["input_ids"].size(1)
        completion_ids = prompt_completion_ids[:, prompt_length:]

        # 获取模型和参考模型的每个token的log概率
        def get_per_token_logps(model, input_ids, num_logits_to_keep):
            logits = model(input_ids, num_logits_to_keep=num_logits_to_keep + 1).logits
            logits = logits[:, :-1, :]

            per_token_logps = []
            for logits_row, input_ids_row in zip(logits, input_ids[:, -num_logits_to_keep:]):
                log_probs = logits_row.log_softmax(dim=-1)
                token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
                per_token_logps.append(token_log_prob)
            return torch.stack(per_token_logps)

        num_logits_to_keep = completion_ids.size(1)
        per_token_logps = get_per_token_logps(model, prompt_completion_ids, num_logits_to_keep)

        with torch.inference_mode():
            if self.ref_model is not None:
                ref_per_token_logps = get_per_token_logps(self.ref_model, prompt_completion_ids, num_logits_to_keep)
            else:
                with self.accelerator.unwrap_model(model).disable_adapter():
                    ref_per_token_logps = get_per_token_logps(model, prompt_completion_ids, num_logits_to_keep)

        # 计算KL散度
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1

        # 处理EOS token之后的mask
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # 解码生成的completions
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = [[{"role": "assistant", "content": completion}] for completion in completions]

        # 计算奖励
        prompts = [prompt for prompt in prompts for _ in range(self.num_generations)]

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, PreTrainedModel):
                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]
                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]
            else:
                reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
                for key in reward_kwargs:
                    for example in inputs:
                        reward_kwargs[key].extend([example[key]] * self.num_generations)
                output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # 计算总奖励和优势
        rewards = rewards_per_func.sum(dim=1)
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        # 计算损失
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

        # 记录指标
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)

        reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())
        self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item())

        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        return loss