import unittest
import torch
from transformers import MistralConfig
from llamafactory.train.mix.mistral_mtp_model import MistralMTPModel, MistralMTPForCausalLM
from transformers import MistralModel

class ExtendedMistralConfig(MistralConfig):
    """扩展的MistralConfig，添加MTP相关配置"""
    def __init__(self, n_future_tokens=1, **kwargs):
        super().__init__(**kwargs)
        self.n_future_tokens = n_future_tokens

class TestMistralMTPModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 使用扩展的配置类
        cls.config = ExtendedMistralConfig(
            vocab_size=32000,
            hidden_size=128,
            intermediate_size=512,
            num_hidden_layers=2,
            num_attention_heads=4,
            max_position_embeddings=128,
            n_future_tokens=2,  # MTP 特定配置
            torch_dtype=torch.float32,  # 使用float32以便于调试
            _attn_implementation="sdpa",  # 使用 PyTorch 原生的 SDPA
        )
        
        # 检查是否有可用的GPU并设置设备
        if torch.cuda.is_available():
            # 强制使用单个 GPU
            torch.cuda.set_device(0)
            cls.device = torch.device("cuda:0")
        else:
            cls.device = torch.device("cpu")
        print(f"\nUsing device: {cls.device}")
        
        # 初始化模型并移到GPU
        cls.model = MistralMTPModel(cls.config).to(cls.device)
        cls.lm_model = MistralMTPForCausalLM(cls.config).to(cls.device)
        
        # 将模型设置为评估模式
        cls.model.eval()
        cls.lm_model.eval()

    def test_model_forward_no_nan(self):
        """测试模型前向传播不产生NaN值"""
        batch_size = 2
        seq_length = 16
        
        # 准备输入数据并移到GPU
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_length)).to(self.device)
        attention_mask = torch.ones_like(input_ids).to(self.device)
        
        # 创建原始 Mistral 模型进行对比
        original_config = MistralConfig(
            vocab_size=self.config.vocab_size,
            hidden_size=self.config.hidden_size,
            intermediate_size=self.config.intermediate_size,
            num_hidden_layers=self.config.num_hidden_layers,
            num_attention_heads=self.config.num_attention_heads,
            num_key_value_heads=self.config.num_attention_heads,  # 确保这个参数也设置正确
            max_position_embeddings=self.config.max_position_embeddings,
            torch_dtype=self.config.torch_dtype,
        )
        print(f"\nOriginal model config: {original_config}")
        
        original_model = MistralModel(original_config).to(self.device)
        original_model.eval()  # 设置为评估模式

        print("\n=== Original Mistral Model ===")
        with torch.no_grad():
            original_outputs = original_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
            print(f"Original model output shape: {original_outputs.last_hidden_state.shape}")

        print("\n=== Our MTP Model ===")
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
            print(f"MTP model output shape: {outputs.last_hidden_state.shape}")
        
        # 检查输出中是否有NaN
        self.assertFalse(
            torch.isnan(outputs.last_hidden_state).any(),
            "发现NaN值在last_hidden_state中"
        )
        
        # 检查输出形状
        expected_shape = (batch_size, self.config.n_future_tokens + 1, seq_length, self.config.hidden_size)
        self.assertEqual(
            outputs.last_hidden_state.shape,
            expected_shape,
            f"输出形状不符合预期: 得到 {outputs.last_hidden_state.shape}, 期望 {expected_shape}"
        )

    def test_lm_forward_no_nan(self):
        """测试语言模型前向传播不产生NaN值"""
        batch_size = 2
        seq_length = 16
        
        # 准备输入数据并移到GPU
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_length)).to(self.device)
        attention_mask = torch.ones_like(input_ids).to(self.device)
        labels = input_ids.clone()
        
        # 运行模型，显式设置use_cache=False
        with torch.no_grad():
            outputs = self.lm_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=False,  # 添加这个参数
            )
        
        # 检查logits中是否有NaN
        self.assertFalse(
            torch.isnan(outputs.logits).any(),
            "发现NaN值在logits中"
        )
        
        # 检查loss是否为NaN
        self.assertFalse(
            torch.isnan(outputs.loss),
            "发现NaN值在loss中"
        )
        
        # 检查logits形状
        expected_shape = (batch_size, self.config.n_future_tokens + 1, seq_length, self.config.vocab_size)
        self.assertEqual(
            outputs.logits.shape,
            expected_shape,
            f"logits形状不符合预期: 得到 {outputs.logits.shape}, 期望 {expected_shape}"
        )

    def test_gradient_flow(self):
        """测试梯度流动是否正常且没有NaN值"""
        batch_size = 2
        seq_length = 16
        
        # 准备输入数据并移到GPU
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_length)).to(self.device)
        attention_mask = torch.ones_like(input_ids).to(self.device)
        labels = input_ids.clone()
        
        # 设置为训练模式
        self.lm_model.train()
        
        # 前向传播，显式设置use_cache=False
        outputs = self.lm_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,  # 添加这个参数
        )
        
        # 反向传播
        outputs.loss.backward()
        
        # 检查所有参数的梯度是否有NaN
        for name, param in self.lm_model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.assertFalse(
                    torch.isnan(param.grad).any(),
                    f"发现NaN值在参数 {name} 的梯度中"
                )
        
        # 恢复评估模式
        self.lm_model.eval()

    def test_different_sequence_lengths(self):
        """测试序列长度为16的输入"""
        batch_size = 2
        seq_length = 16  # 固定序列长度
        
        # 准备输入数据并移到GPU
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_length)).to(self.device)
        
        # 创建正确的 attention_mask
        attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long).to(self.device)
        
        # 创建正确的 position_ids
        position_ids = torch.arange(seq_length, dtype=torch.long).unsqueeze(0).expand(batch_size, -1).to(self.device)
        
        print(f"\nTesting sequence length: {seq_length}")
        print(f"input_ids shape: {input_ids.shape}")
        print(f"attention_mask shape: {attention_mask.shape}")
        print(f"position_ids shape: {position_ids.shape}")
        
        # 运行模型，显式设置use_cache=False
        with torch.no_grad():
            outputs = self.lm_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
            )
        
        # 检查输出形状
        expected_logits_shape = (batch_size, self.config.n_future_tokens + 1, seq_length, self.config.vocab_size)
        self.assertEqual(
            outputs.logits.shape,
            expected_logits_shape,
            f"logits形状不符合预期: 得到 {outputs.logits.shape}, 期望 {expected_logits_shape}"
        )
        
        # 检查输出是否有NaN
        self.assertFalse(
            torch.isnan(outputs.logits).any(),
            f"发现NaN值在logits中"
        )
        
        # 检查输出是否有无穷值
        self.assertTrue(
            torch.isfinite(outputs.logits).all(),
            f"发现无穷值在logits中"
        )

if __name__ == '__main__':
    unittest.main() 