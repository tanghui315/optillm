# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# great reference: https://github.com/vllm-project/vllm/issues/11400


import time

from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import requests


class RemoteModel:
    """
    launch with:
    export LD_LIBRARY_PATH=$(python -c "import site; print(site.getsitepackages()[0] + '/nvidia/nvjitlink/lib')"):$LD_LIBRARY_PATH
    python3 -m sglang.launch_server --model-path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --port=30010 --skip-tokenizer-init --mem-fraction-static 0.4
    python3 -m sglang.launch_server --model-path HuggingFaceTB/SmolLM2-135M-Instruct --port=30010 --skip-tokenizer-init --mem-fraction-static 0.4 --host=0.0.0.0

    # on a separate node
    python3 -m sglang.launch_server --model-path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --port=30010 --skip-tokenizer-init --mem-fraction-static 0.7 --host=0.0.0.0 --dp-size=8

    python3 -m sglang.launch_server --model-path HuggingFaceTB/SmolLM2-1.7B-Instruct --port=30010 --skip-tokenizer-init --mem-fraction-static 0.6 --host=0.0.0.0 --dp-size=8


    python3 -m sglang.launch_server --model-path open-r1/Qwen2.5-Coder-7B-Instruct-SFT --revision v00.08-step-000001280  --port=30010 --skip-tokenizer-init --mem-fraction-static 0.7 --host=0.0.0.0 --dp-size=8
    """

    def __init__(self, remote_model_url, remote_model_port, tokenizer=None, stop_token_id=None):
        self.remote_model_url = remote_model_url
        self.remote_model_port = remote_model_port
        self.tokenizer = tokenizer
        self.stop_token_id = stop_token_id

    def is_healthy(self, timeout=5):
        """Checks if the remote model server is up and running."""
        try:
            url = f"http://{self.remote_model_url}:{self.remote_model_port}/health"
            response = requests.get(url, timeout=timeout)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def wait_for_server(self, max_retries=120, delay=5):
        """Waits for the server to become available before proceeding."""
        for attempt in range(max_retries):
            if self.is_healthy():
                print("Remote model server is healthy!")
                return True
            print(f"Waiting for server to start... (Attempt {attempt + 1}/{max_retries})")
            time.sleep(delay)
        raise RuntimeError("Remote model server did not start in time.")

    def generate_bak(
        self, prompts: list[str], max_new_tokens=256, temperature=0.8, num_generations=1
    ) -> list[dict]:
        """
        生成文本并返回结果，包括生成的文本、token IDs和log概率。
        
        Args:
            prompts: 文本提示列表
            max_new_tokens: 最大生成的新token数量
            temperature: 采样温度
            num_generations: 每个提示生成的回复数量
            
        Returns:
            包含生成结果的字典列表
        """

        # 准备请求体
        request_body = {
            "text": prompts,
            "sampling_params": {
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
                "stop_token_ids": [self.stop_token_id] if self.stop_token_id else None,
                "n": num_generations,
            },
            "stream": False,
        }
        
        # 发送POST请求到服务器
        response = requests.post(
            f"http://{self.remote_model_url}:{self.remote_model_port}/generate", json=request_body
        )
        
        # 添加调试信息
        # print(f"Response状态码: {response.status_code}")
        # print(f"Response内容预览: {response.text[:500]}")  # 只显示前500个字符
        
        try:
            response_json = response.json()
            # 添加详细的响应结构调试信息
            # print(f"\n===== 详细响应结构调试信息 =====")
            # print(f"响应JSON类型: {type(response_json)}")
            
            # if isinstance(response_json, list):
            #     print(f"响应是列表，包含 {len(response_json)} 个元素")
            #     if len(response_json) > 0:
            #         print(f"第一个结果结构:")
            #         for key, value in response_json[0].items():
            #             if isinstance(value, dict):
            #                 print(f"  - {key} (字典): {list(value.keys())}")
            #             elif isinstance(value, list) and len(value) > 0:
            #                 print(f"  - {key} (列表，长度 {len(value)}): 第一个元素类型 {type(value[0])}")
            #             else:
            #                 print(f"  - {key} ({type(value)}): {str(value)[:50]}{'...' if len(str(value)) > 50 else ''}")
            # else:
            #     print(f"响应是字典，包含以下键: {list(response_json.keys())}")
            # print(f"===== 调试信息结束 =====\n")
        except Exception as e:
            print(f"解析JSON时出错: {e}")
            return []

        results = []

        for i, result in enumerate(response_json):
            # 获取生成的文本
            generated_text = result.get("text", "")
            
            # 使用tokenizer将文本转换为token IDs
            completion_ids = self.tokenizer.encode(generated_text, add_special_tokens=False)
            
            # 构建输出结果
            output = {
                "text": generated_text,
                "completion_ids": completion_ids
            }
            results.append(output)

        return results
    
    def generate(
        self, input_ids: list[list[int]], max_new_tokens=256, temperature=0.8, num_generations=2
    ) -> tuple[list[list[int]], list[list[int]]]:
        # Prepare the request body
        request_body = {
            "input_ids": input_ids,
            "sampling_params": {
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
                "stop_token_ids": [self.stop_token_id],
                "n": num_generations,
            },
            "stream": False,
            "return_logprob": True,
            "logprob_start_len": 0,
        }

        # Send the POST request to the server
        # add a few retries?
        response = requests.post(
            f"http://{self.remote_model_url}:{self.remote_model_port}/generate", json=request_body
        )
        response_json = response.json()

        examples = []

        for i, result in enumerate(response_json):
            prompt_index = i // num_generations
            prompt_ids = input_ids[prompt_index]
            completion_ids = result["token_ids"]
            prompt_log_probs = [prob[0] for prob in result["meta_info"]["input_token_logprobs"]]
            completion_log_probs = [prob[0] for prob in result["meta_info"]["output_token_logprobs"]]

            example = {
                "prompt_ids": prompt_ids,
                "completion_ids": completion_ids,
                "prompt_log_probs": prompt_log_probs,
                "completion_log_probs": completion_log_probs,
            }
            examples.append(example)

        return examples
    

    def load_weights_from_path(self, path: str):
        url = f"http://{self.remote_model_url}:{self.remote_model_port}/update_weights_from_disk"
        data = {"model_path": path}

        response = requests.post(url, json=data)
        print(response.text)
        assert response.json()["success"] is True
        # assert response.json()["message"] == "Succeeded to update model weights."
        assert response.json().keys() == {"success", "message"}


if __name__ == "__main__":
    from datasets import load_dataset

    url = "0.0.0.0"
    port = 30010
    MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    remote_model = RemoteModel(url, port, tokenizer)
    dataset = load_dataset("AI-MO/NuminaMath-TIR", split="train")
    dataloader = DataLoader(dataset, batch_size=4)

    for i, batch in zip(range(2), dataloader):
        problems = batch["problem"]
        ids = tokenizer(problems)
        new_ids, logprobs = remote_model.generate(ids["input_ids"])
        print(new_ids)
        print(logprobs)
