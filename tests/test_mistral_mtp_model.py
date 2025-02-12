import unittest
import torch
from transformers import MistralConfig
from llamafactory.train.mix.mistral_mtp_model import MistralMTPForCausalLM

class TestMistralMTPModel(unittest.TestCase):
    def setUp(self):
        # 创建最小化配置
        self.config = MistralConfig(
            vocab_size=100,  # 减少词表大小
            hidden_size=64,  # 原版4096
            intermediate_size=128,  # 原版14336
            num_hidden_layers=2,  # 原版32
            num_attention_heads=2,  # 原版32
            num_key_value_heads=1,
            max_position_embeddings=64,
        )
        setattr(self.config, "n_future_tokens", 2)
        self.model = MistralMTPForCausalLM(self.config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def _create_test_batch(self, seq_lengths):
        """创建测试批次数据"""
        batch_size = len(seq_lengths)
        max_len = max(seq_lengths)
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, max_len)).to(self.device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long).to(self.device)
        
        # 创建有效attention mask
        for i, l in enumerate(seq_lengths):
            attention_mask[i, :l] = 1
        
        # 创建labels（最后一个token为-100）
        labels = input_ids.clone()
        for i, l in enumerate(seq_lengths):
            if l > 0:
                labels[i, l-1] = -100
                
        return input_ids, attention_mask, labels

    def test_normal_sequence(self):
        """测试正常长度序列"""
        input_ids, attention_mask, labels = self._create_test_batch([8, 8])
        outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
        
        # 验证基本属性
        self.assertIsNotNone(outputs.loss)
        self.assertTrue(isinstance(outputs.loss, torch.Tensor))
        self.assertGreater(outputs.loss.item(), 0)  # 正常情况loss应大于0
        
        # 验证logits的形状和内容
        self.assertEqual(outputs.logits.shape[1], self.config.n_future_tokens + 1,  # 验证预测头数量
                        "logits应该包含所有预测头的输出")
        self.assertFalse(torch.isnan(outputs.logits).any(), "logits不应包含NaN")
        self.assertFalse(torch.isinf(outputs.logits).any(), "logits不应包含Inf")

    def test_short_sequence(self):
        """测试短序列"""
        input_ids, attention_mask, labels = self._create_test_batch([3, 4])
        outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
        
        # 验证loss计算
        self.assertIsNotNone(outputs.loss)
        self.assertFalse(torch.isnan(outputs.loss), "loss不应为NaN")
        self.assertGreater(outputs.loss.item(), 0, "loss应大于0")

    def test_edge_case(self):
        """测试边界情况（序列很短）"""
        input_ids, attention_mask, labels = self._create_test_batch([2, 1])
        outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
        
        # 即使是边界情况，模型也应该能正常运行
        self.assertIsNotNone(outputs.logits)
        if outputs.loss is not None:  # 如果有足够长度计算loss
            self.assertFalse(torch.isnan(outputs.loss), "loss不应为NaN")

    def test_loss_calculation(self):
        """测试损失计算正确性"""
        batch_size = 2
        seq_len = 5
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len)).to(self.device)
        labels = torch.full((batch_size, seq_len), -100).to(self.device)
        labels[:, 2:4] = 1  # 设置中间有效目标
        
        outputs = self.model(input_ids, labels=labels)
        
        # 验证loss
        self.assertIsNotNone(outputs.loss)
        self.assertGreater(outputs.loss.item(), 0, "有效目标的loss应大于0")
        
        # 验证logits
        expected_shape = (batch_size, self.config.n_future_tokens + 1, seq_len, self.config.vocab_size)
        self.assertEqual(outputs.logits.shape, expected_shape, 
                        f"logits形状应为{expected_shape}")

if __name__ == "__main__":
    unittest.main()