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

    def test_projection_stability(self):
        """测试 projection 层的数值稳定性"""
        # 创建极端值输入
        batch_size = 2
        seq_len = 16
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len)).to(self.device)
        
        # 1. 测试正常范围输入
        outputs = self.model(input_ids)
        self.assertFalse(torch.isnan(outputs.logits).any(), "正常输入下不应有NaN")
        
        # 2. 测试大值输入
        large_embeds = self.model.get_input_embeddings()(input_ids) * 1000
        outputs = self.model(inputs_embeds=large_embeds)
        self.assertFalse(torch.isnan(outputs.logits).any(), "大值输入下不应有NaN")
        
        # 3. 测试小值输入
        small_embeds = self.model.get_input_embeddings()(input_ids) * 0.001
        outputs = self.model(inputs_embeds=small_embeds)
        self.assertFalse(torch.isnan(outputs.logits).any(), "小值输入下不应有NaN")
        
        # 4. 测试混合极端值
        mixed_embeds = self.model.get_input_embeddings()(input_ids)
        mixed_embeds[:, :5] = 1000  # 前5个位置设为大值
        mixed_embeds[:, 5:10] = 0.001  # 接下来5个位置设为小值
        outputs = self.model(inputs_embeds=mixed_embeds)
        self.assertFalse(torch.isnan(outputs.logits).any(), "混合极端值输入下不应有NaN")

    def test_projection_gradient(self):
        """测试 projection 层的梯度稳定性"""
        batch_size = 2
        seq_len = 16
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len)).to(self.device)
        labels = input_ids.clone()
        labels[:, -1] = -100  # 设置最后一个token为目标
        
        # 启用梯度计算
        self.model.train()
        
        # 1. 测试正常梯度
        outputs = self.model(input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        
        # 检查projection层的梯度
        for i, module in enumerate(self.model.model.mtp_modules):
            for name, param in module['projection'].named_parameters():
                self.assertFalse(torch.isnan(param.grad).any(), 
                               f"MTP模块{i}的projection层{name}存在NaN梯度")
                self.assertFalse(torch.isinf(param.grad).any(), 
                               f"MTP模块{i}的projection层{name}存在Inf梯度")
                
        # 2. 测试梯度范围
        max_grad = max(p.grad.abs().max() for p in self.model.parameters() if p.grad is not None)
        self.assertLess(max_grad, 1000, "梯度值不应过大")

    def test_projection_intermediate_values(self):
        """测试 projection 层的中间值"""
        batch_size = 2
        seq_len = 16
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len)).to(self.device)
        
        # 注册钩子来捕获中间值
        intermediate_values = {}
        def hook_fn(name):
            def hook(module, input, output):
                intermediate_values[name] = output.detach()
            return hook
        
        # 为每个MTP模块的projection层注册钩子
        hooks = []
        for i, module in enumerate(self.model.model.mtp_modules):
            for j, layer in enumerate(module['projection']):
                hook = layer.register_forward_hook(hook_fn(f'mtp_{i}_layer_{j}'))
                hooks.append(hook)
        
        # 前向传播
        outputs = self.model(input_ids)
        
        # 检查所有中间值
        for name, value in intermediate_values.items():
            self.assertFalse(torch.isnan(value).any(), f"{name}中存在NaN")
            self.assertFalse(torch.isinf(value).any(), f"{name}中存在Inf")
            
            # 检查数值范围
            max_val = value.abs().max().item()
            self.assertLess(max_val, 1000, f"{name}的值范围过大: {max_val}")
            
            # 打印统计信息
            print(f"\n{name} 统计信息:")
            print(f"- 最小值: {value.min().item():.4f}")
            print(f"- 最大值: {value.max().item():.4f}")
            print(f"- 均值: {value.mean().item():.4f}")
            print(f"- 标准差: {value.std().item():.4f}")
        
        # 移除钩子
        for hook in hooks:
            hook.remove()

    def test_sequence_length_impact(self):
        """测试不同序列长度对projection层的影响"""
        batch_size = 2
        for seq_len in [8, 16, 32, 48, 64]:  # 测试不同长度
            input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len)).to(self.device)
            
            try:
                outputs = self.model(input_ids)
                self.assertFalse(torch.isnan(outputs.logits).any(), 
                               f"序列长度{seq_len}时出现NaN")
            except Exception as e:
                self.fail(f"序列长度{seq_len}时发生错误: {str(e)}")

if __name__ == "__main__":
    unittest.main()