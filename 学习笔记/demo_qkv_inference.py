"""
演示：从分词 → Embedding → QKV → Attention → 推理生成

用一个小型随机模型展示完整链路，理解概念用，不依赖 GPU。

运行方式：
    python 学习笔记/demo_qkv_inference.py

前置知识（按顺序读）：
    1. show_qkv_origin()     — Q、K、V 到底从哪里来
    2. simulate_training()   — 训练做了什么，W_Q/W_K/W_V 怎么来的
    3. inference_demo()      — 推理时有无 KV Cache 的区别

相关文档：
    学习笔记/LLM推理基础-从训练到推理.md
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 第 0 部分：搭建一个微型语言模型
# ============================================================================

class TinyAttention(nn.Module):
    """
    单头 Self-Attention —— QKV 的核心计算单元。

    这层做的事情（按调用顺序）：
        ① 接收 embedding 向量 x，分别过 W_Q、W_K、W_V 三个矩阵 → 得到 Q、K、V
        ② 如果传入了 kv_cache（历史缓存的 K、V），拼接到当前 K、V 前面
        ③ Q 和 K 做点积 → 得到注意力分数（哪些 token 和我关系大）
        ④ 除以 sqrt(d) 缩放 → 防止分数太大导致 softmax 梯度消失
        ⑤ 用 causal mask 遮掉"未来"的 token → 保证不自欺欺人
        ⑥ softmax 把分数变成概率 → 注意力权重
        ⑦ 用权重对 V 加权求和 → 输出融合了上下文信息的向量
    """

    def __init__(self, hidden_dim: int = 64):
        """
        Args:
            hidden_dim: 隐藏层维度，即每个 token 用多长的向量表示。
                        真实模型通常是 4096（Llama-7B）到 8192（Llama-70B）。
        """
        super().__init__()

        # ---------------------------------------------------------------
        # W_Q, W_K, W_V —— 这三个矩阵是整个 Attention 仅有的可学习参数。
        #
        # 它们的作用是：把同一个 embedding 向量，投影到三个不同的"语义空间"：
        #   W_Q → "提问空间"：这个 token 想从其他 token 那里了解什么？
        #   W_K → "匹配空间"：这个 token 能提供什么信息给其他 token？
        #   W_V → "内容空间"：这个 token 实际携带的信息是什么？
        #
        # 为什么同一个向量要乘三个不同的矩阵？
        #   举例：token "苹果"
        #     作为 Q："我想知道苹果是什么颜色的？"     → 关注"颜色"相关词
        #     作为 K："有人问水果，我这里有苹果的信息"  → 被"水果"相关词匹配
        #     作为 V："苹果 = 红色、甜的、可以吃的水果" → 提供具体信息
        #   三个矩阵让同一个 token 在不同角色下有不同的"表达方式"。
        #
        # 训练时会不断调整这三个矩阵里的数值，训完就冻结不再改变。
        # ---------------------------------------------------------------
        self.W_Q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_K = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_V = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        前向传播：输入 embedding 向量 → 输出融合了上下文的向量。

        Args:
            x: 输入向量，形状 [batch, seq_len, hidden_dim]
               - batch: 一次处理几个请求（通常为 1）
               - seq_len: 这一步输入了几个 token
                 * 无 KV Cache 时：每次都传全部历史 token（3, 4, 5, ...）
                 * 有 KV Cache 时：第 1 步传全部 prompt，之后每次只传 1 个新 token
               - hidden_dim: 每个 token 的向量维度
            kv_cache: 之前步骤缓存的 (K, V)，格式为 (K_past, V_past)
                      - 第 1 步时为 None（还没有缓存）
                      - 后续步骤传入之前累积的全部 K、V

        Returns:
            output: 注意力输出，形状 [batch, seq_len, hidden_dim]
            new_kv: 更新后的 (K, V)，包含历史缓存 + 本步新增
        """
        # ================================================================
        # 步骤 ①：计算 Q、K、V
        #
        # 输入 x 是从 embedding 查表得到的向量（比如 [1.2, -0.5, 0.8, ...]）
        # 分别乘三个训练好的矩阵 → 得到三个不同用途的向量
        #
        # 举例（hidden_dim=4，数值是示意）：
        #   x = [0.5, 0.3, -0.2, 0.1]      ← "苹果"的 embedding
        #   Q = x × W_Q = [0.8, 0.1, 0.3, -0.5]   ← "苹果"作为提问者
        #   K = x × W_K = [-0.2, 0.7, 0.4, 0.1]   ← "苹果"作为被匹配者
        #   V = x × W_V = [0.3, 0.9, 0.2, -0.1]   ← "苹果"的实际内容
        # ================================================================
        Q = self.W_Q(x)  # [batch, seq_len, hidden_dim] — 当前 token(s) 的 Query
        K = self.W_K(x)  # [batch, seq_len, hidden_dim] — 当前 token(s) 的 Key
        V = self.W_V(x)  # [batch, seq_len, hidden_dim] — 当前 token(s) 的 Value

        # ================================================================
        # 步骤 ②：拼接历史 KV Cache（如果有的话）
        #
        # 这是 KV Cache 的核心操作：把之前算好的 K、V 拼到当前新算的前面。
        # 拼接后，Attention 计算时能看到"全部历史 + 当前"的完整上下文。
        #
        # 为什么 K 和 V 能缓存，Q 不能？
        #   - K 和 V 取决于"这个 token 本身是什么"，一旦生成就永远不变了
        #   - Q 取决于"当前正在生成的 token"，每次都不一样
        #   - 所以 Q 每次必须重算，K 和 V 算一次就够了
        # ================================================================
        if kv_cache is not None:
            K_past, V_past = kv_cache  # 取出之前缓存的全部历史 K 和 V
            # 沿着 seq_len 维度拼接：历史的在前，当前新增的在后
            K = torch.cat([K_past, K], dim=1)  # [batch, past_len + new_len, hidden_dim]
            V = torch.cat([V_past, V], dim=1)  # [batch, past_len + new_len, hidden_dim]

        # 把新的完整 KV 打包返回，供下一步使用
        new_kv = (K, V)

        # ================================================================
        # 步骤 ③-④：计算注意力分数 + 缩放
        #
        # scores = Q × K^T / sqrt(d_k)
        #
        # Q × K^T 的含义：每个查询 token（Q）和每个被查 token（K）的点积。
        # 点积越大 → 两个向量方向越一致 → 关系越密切。
        #
        # 举例（Attention 分数矩阵，行为 Q，列为 K）：
        #            K₀("我")  K₁("爱")  K₂("北京")  K₃("天")
        #   Q₀("我")  [ 0.9      0.3      0.2       0.1  ]  ← "我"最关注"我"
        #   Q₁("爱")  [ 0.4      0.8      0.3       0.1  ]  ← "爱"最关注"爱"
        #   Q₂("北京")[ 0.3      0.5      0.9       0.2  ]  ← "北京"最关注"北京"
        #   Q₃("天")  [ 0.2      0.3      0.6       0.8  ]  ← "天"最关注"天"和"北京"
        #
        # 除以 sqrt(d_k) 是为了防止点积值太大。
        # 如果 d_k=4096，点积可能到几百 → softmax 后几乎全是 0 或 1 → 梯度消失。
        # ================================================================
        seq_len = Q.shape[1]     # 当前传入了几个 token 的 Q
        kv_len = K.shape[1]      # 总共有多少个 token 的 K（历史 + 当前）
        d_k = Q.shape[2]         # hidden_dim

        # Q @ K^T → [batch, seq_len, kv_len]
        scores = (Q @ K.transpose(-2, -1)) / (d_k ** 0.5)

        # ================================================================
        # 步骤 ⑤：Causal Mask（因果遮罩）
        #
        # 语言模型是"自回归"的：只能根据已生成的 token 预测下一个。
        # 所以在计算 Attention 时，不能让第 i 个 token "偷看"第 i+1 个及之后的。
        #
        # 具体做法：用一个上三角矩阵把"未来"位置的分数设为 -inf。
        # softmax(-inf) = 0 → 注意力权重为 0 → 等于没看。
        #
        # 可视化（√ 允许看，✕ 遮住）：
        #         K₀  K₁  K₂  K₃
        #   Q₀  [ √   ✕   ✕   ✕ ]  ← Q₀ 只能看 K₀
        #   Q₁  [ √   √   ✕   ✕ ]  ← Q₁ 能看 K₀,K₁
        #   Q₂  [ √   √   √   ✕ ]  ← Q₂ 能看 K₀,K₁,K₂
        #   Q₃  [ √   √   √   √ ]  ← Q₃ 能看所有
        # ================================================================
        # 生成上三角 mask：对角线上移 offset 格
        # offset = 1 + kv_len - seq_len 确保了 mask 对齐到"当前 token 能看到多少历史"
        mask = torch.triu(
            torch.ones(seq_len, kv_len),
            diagonal=1 + kv_len - seq_len,
        )
        mask = mask.bool().to(scores.device)
        scores = scores.masked_fill(mask, float("-inf"))  # 把"未来"位置设为 -inf

        # ================================================================
        # 步骤 ⑥：Softmax → 将分数变成权重（概率分布）
        #
        # softmax 做的事：
        #   1. 对每个分数取 e 的幂（让负数变成正小数，让大数变得更大）
        #   2. 除以总和（让所有权重加起来等于 1）
        #
        # 举例：scores = [2.0, 1.0, 0.5, -inf]
        #   e^2.0=7.39, e^1.0=2.72, e^0.5=1.65, e^(-inf)=0
        #   总和 = 11.76
        #   权重 = [0.63, 0.23, 0.14, 0.00]  ← 63% 注意力在第一个 token 上
        # ================================================================
        attn_weights = F.softmax(scores, dim=-1)  # [batch, seq_len, kv_len]

        # ================================================================
        # 步骤 ⑦：用注意力权重对 V 加权求和
        #
        # 每个 token 的输出 = 所有 token 的 V 按注意力权重加权平均。
        #
        # 延续上面的例子：
        #   权重 = [0.63, 0.23, 0.14, 0.00]
        #   V₀ = "我"的内容, V₁ = "爱"的内容, V₂ = "北京"的内容, V₃ = "天"的内容
        #   输出 = 0.63×V₀ + 0.23×V₁ + 0.14×V₂ + 0.00×V₃
        #       ≈ 主要来自"我"，掺杂一点"爱"和"北京"
        #
        # 这就是 Attention 的本质：
        #   "我当前这个位置，应该从每个历史位置吸收多少信息？"
        # ================================================================
        output = attn_weights @ V  # [batch, seq_len, hidden_dim]

        return output, new_kv


class TinyModel(nn.Module):
    """
    微型语言模型：Embedding → Attention → 输出投影

    模型架构（三层）：
        token IDs
           ↓
        Embedding     ← 把整数 ID 查表变成浮点数向量
           ↓
        Attention     ← 让每个 token 看到上下文（包含 QKV 计算）
           ↓
        LM Head       ← 把向量映射回词汇表大小，得到每个词的概率
           ↓
        logits        ← 取 argmax 就知道下一个最可能的 token
    """

    def __init__(self, vocab_size: int = 100, hidden_dim: int = 64):
        """
        Args:
            vocab_size: 词汇表大小。真实模型通常是 32000~150000。
                       这里用 100 只是为了演示，降低随机性。
            hidden_dim: 隐藏层维度。每个 token 用多长的向量表示。
        """
        super().__init__()

        # ---------------------------------------------------------------
        # Embedding 层（嵌入表）
        #
        # 本质就是一个大表格：行 = 词汇表大小，列 = hidden_dim。
        # 输入 token ID → 查表 → 返回该 token 对应的向量。
        #
        # 比如 vocab_size=100, hidden_dim=64 → 100×64 的表格
        #   token ID 0  → [ 0.12, -0.34,  0.78, ...] (64 个数)
        #   token ID 1  → [ 0.45,  0.21, -0.11, ...] (64 个数)
        #   ...
        #
        # 这个表的值也是在训练中学出来的。
        # 语义相近的词，学出来的向量也相近（比如"苹果"和"香蕉"的向量距离近）。
        # ---------------------------------------------------------------
        self.embed = nn.Embedding(vocab_size, hidden_dim)

        # QKV + Attention 计算
        self.attention = TinyAttention(hidden_dim)

        # ---------------------------------------------------------------
        # LM Head（语言模型头）
        #
        # 把 hidden_dim 维的向量映射回 vocab_size 维。
        # 输出向量的每个位置代表对应 token 的"分数"（logit）。
        # 分数最高的那个 token 就是模型认为最可能的下一个词。
        # ---------------------------------------------------------------
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        token_ids: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        模型前向传播。

        Args:
            token_ids: [batch, seq_len] token ID 序列
                       - 训练时：seq_len = 整句话的长度
                       - 无缓存推理时：seq_len = 1, 2, 3, ...（不断增长）
                       - 有缓存推理时：第 1 步 = prompt 长度，之后 = 1
            kv_cache: 之前累积的 (K, V)，第一步为 None

        Returns:
            logits: [batch, seq_len, vocab_size] 每个位置对下一个 token 的预测分数
            kv_cache: 更新后的 (K, V)，传入下一步
        """
        # ① Token ID → 查嵌入表 → 浮点数向量
        x = self.embed(token_ids)  # [batch, seq_len, hidden_dim]

        # ② Embedding 向量 → Attention（含 QKV 计算 + 上下文融合 + KV 缓存管理）
        x, new_kv = self.attention(x, kv_cache)

        # ③ 融合后的向量 → 映射回词汇表大小 → logits
        logits = self.lm_head(x)  # [batch, seq_len, vocab_size]

        return logits, new_kv


# ============================================================================
# 第 1 部分：模拟训练 —— W_Q、W_K、W_V 是如何"学会"的
# ============================================================================

def simulate_training() -> TinyModel:
    """
    模拟训练过程，展示 W_Q、W_K、W_V 是怎么被调整的。

    真实训练做的事情（简化版）：
        1. 拿一堆文本，切成"输入-标签"对
           输入: "中国的首都是"
           标签: "中国的首都是北京"  ← 让模型预测每个位置的下一个词

        2. 前向传播（forward）
           整句话一次性喂给模型 → 得到每个位置的预测

        3. 计算损失（loss）
           模型的预测 vs 正确答案 → 差距有多大

        4. 反向传播（backward）
           从 loss 往回算梯度 → 每个参数是调大还是调小能减少误差？

        5. 更新参数（optimizer.step()）
           按梯度的方向微调 W_Q、W_K、W_V 等所有参数

        6. 重复 1-5，几万亿次 → 参数收敛 → 保存模型文件

    训练 vs 推理的关键区别：
        ┌──────────┬─────────────────┬─────────────────┐
        │          │ 训练            │ 推理             │
        ├──────────┼─────────────────┼─────────────────┤
        │ 输入方式 │ 整句一次性给     │ 逐 token 生成    │
        │ 前向传播 │ 有               │ 有               │
        │ 反向传播 │ 有（调参数）     │ 无（参数冻结）   │
        │ W_Q/W_K/W_V │ 一直在变    │ 永远不变         │
        │ 输出      │ 不需要，只要 loss │ 需要，输出 token │
        └──────────┴─────────────────┴─────────────────┘
    """
    model = TinyModel(vocab_size=100, hidden_dim=64)
    # Adam 优化器：负责根据梯度调整参数
    # model.parameters() 包含了 W_Q、W_K、W_V、Embedding 表、LM Head 等所有可学参数
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    print("【模拟训练】（5 步随机迭代）")
    print("─" * 45)
    for step in range(5):
        # ---- 1. 构造训练数据 ----
        # 真实场景：从语料库取一句话，这里用随机 token 序列代替
        seq = torch.randint(0, 100, (1, 8))  # [batch=1, seq_len=8]

        # ---- 2. 前向传播 ----
        # 整句话一次性送进去（训练不需要 KV Cache，因为不逐 token 生成）
        logits, _ = model(seq)
        # logits 形状：[1, 8, 100]
        #   → 1 句话，8 个位置，每个位置输出 100 个词的分数

        # ---- 3. 计算损失 ----
        # 真实场景：logits[i] 预测的就是 seq[i+1]，和实际 seq[i+1] 做交叉熵
        # 这里简化为取所有 logits 的均值（仅演示反向传播流程）
        loss = logits.mean()

        # ---- 4. 反向传播 ----
        # PyTorch 自动计算每个参数对 loss 的梯度
        # 比如：∂loss/∂W_Q = ?（W_Q 调大一点 loss 会变大还是变小？）
        optimizer.zero_grad()   # 清空上一步的梯度
        loss.backward()         # 从 loss 往回传播，算出所有参数的梯度

        # ---- 5. 更新参数 ----
        # optimizer 根据梯度微调参数
        # 这一步结束，W_Q、W_K、W_V 就变了一点点
        optimizer.step()

        print(f"  step {step + 1}: loss = {loss.item():.4f} → backward → W 参数已更新")

    print("  训练完成！W_Q、W_K、W_V 现在固定，不再改变。")
    print("  模型文件保存到磁盘 → 推理时直接加载 → 不再做反向传播。\n")
    return model


# ============================================================================
# 第 2 部分：推理 —— 对比有无 KV Cache 的区别
# ============================================================================

def inference_demo(model: TinyModel):
    """
    推理演示：用同一个模型，分别以"无缓存"和"有缓存"两种方式生成 token。

    对比维度：
        ① 每次传给 model 的 token 数量
        ② 每次实际计算了几组新的 QKV
        ③ Attention 最终看到了多少个 token（两者相同！）

    关键结论：
        Attention 看到的上下文一样多，但 KV Cache 方式传给 model
        的输入更少 → 重复计算更少 → 更快。
    """
    print("=" * 55)
    print("【推理演示】")
    print("  输入 prompt: '我 爱 北京'")
    print("  对应 token IDs: [10, 20, 30]")
    print("=" * 55)

    # 模拟一个 prompt："我爱北京" 被分词为 3 个 token
    prompt = torch.tensor([[10, 20, 30]])  # [batch=1, seq_len=3]

    # ====================================================================
    # 方式 A：无 KV Cache
    #
    # 每次把"全部历史 token + 新 token"重新传进 model。
    # model 内部每次都会重新计算所有 token 的 Q、K、V。
    #
    # 流程示意：
    #   第 1 步: model([10, 20, 30])           → 算 3 组 QKV → 输出 [55]
    #   第 2 步: model([10, 20, 30, 55])       → 算 4 组 QKV（前 3 组重复算！）
    #   第 3 步: model([10, 20, 30, 55, 18])   → 算 5 组 QKV（前 4 组重复算！）
    #   ...
    #   第 100 步: model([10, 20, ..., 前面 99 个]) → 算 100+ 组 QKV
    # ====================================================================

    print("\n── 方式 A：无 KV Cache ──")
    print("  策略：每次传入全部历史 token")
    print()

    current_seq = prompt.clone()  # [1, 3] — 当前要传的完整序列
    for step in range(3):
        # 传整句：所有 token 的 Q、K、V 都重新算一遍
        logits, _ = model(current_seq)

        # logits[0, -1] 取最后一个位置的输出（它预测下一个 token）
        # argmax 取分数最高的那个词的 ID
        next_token = logits[0, -1].argmax().item()

        n_input = current_seq.shape[1]  # 传入了几个 token
        print(f"  第 {step + 1} 步:")
        print(f"    传入 token 数: {n_input}")
        print(f"    新计算 QKV 组数: {n_input}")
        print(f"    Attention 看到: {n_input} 个 token（全部来自重算）")
        print(f"    输出的新 token: [{next_token}]")

        # 把新 token 拼到序列末尾 → 下一步序列更长
        current_seq = torch.cat([current_seq, torch.tensor([[next_token]])], dim=1)

    print(f"\n  → 3 步总共计算了 3+4+5 = 12 组 QKV")
    print(f"  → 问题：序列越长，重复计算越多，越来越慢。")
    print(f"  → 生成 N 个 token 的计算量：O(N²)")

    # ====================================================================
    # 方式 B：有 KV Cache
    #
    # 第一步传整个 prompt（和 A 一样），之后每次只传 1 个新 token。
    # 历史的 K、V 存在缓存里，Attention 内部拼接 → 看到的还是完整上下文。
    #
    # 流程示意：
    #   第 1 步: model([10,20,30], cache=None)   → 算 3 组 QKV → 存缓存 → 输出 [55]
    #   第 2 步: model([55], cache=(K_0:2, V_0:2)) → 算 1 组 QKV → 拼缓存 → 看到 4 个 token
    #   第 3 步: model([18], cache=(K_0:3, V_0:3)) → 算 1 组 QKV → 拼缓存 → 看到 5 个 token
    #   ...
    #   第 100 步: model([x], cache=(K_0:99, V_0:99)) → 算 1 组 QKV → 看到 100+ token
    # ====================================================================

    print("\n── 方式 B：有 KV Cache（vLLM 的方式）──")
    print("  策略：第 1 步传全部 prompt，之后每次只传 1 个新 token")
    print()

    kv_cache = None                      # 缓存初始为空
    current_token = prompt.clone()       # [1, 3] — 第一步传全部 prompt

    for step in range(3):
        # 传入当前 token + 缓存
        # Attention 内部会拼接：当前 K,V 拼到缓存的 K,V 后面
        logits, kv_cache = model(current_token, kv_cache=kv_cache)

        next_token = logits[0, -1].argmax().item()

        n_input = current_token.shape[1]                    # 传入了几个 token
        total_seen = kv_cache[0].shape[1]                   # 缓存里总共有几个 token 的 KV
        new_computed = n_input                              # 新算了几组 KV

        if step == 0:
            print(f"  第 {step + 1} 步 (prefill 阶段):")
            print(f"    传入 token 数: {n_input}（整个 prompt）")
            print(f"    新计算 KV 组数: {new_computed}")
            print(f"    KV 缓存后大小: {total_seen} 组")
            print(f"    Attention 看到: {total_seen} 个 token")
            print(f"    输出的新 token: [{next_token}]")
            print(f"    ── 这一步和有缓存的方式 A 一样 ──")
        else:
            print(f"  第 {step + 1} 步 (decode 阶段):")
            print(f"    传入 token 数: {n_input}（只传新 token）")
            print(f"    新计算 KV 组数: {new_computed}")
            print(f"    从缓存读取: {total_seen - new_computed} 组")
            print(f"    Attention 看到: {total_seen} 个 token（缓存 {total_seen - new_computed} + 新 {new_computed}）")
            print(f"    输出的新 token: [{next_token}]")

        # ★ 关键！覆盖 current_token，下一步只传 1 个新 token
        # 不是拼接，而是替换——因为历史的 KV 已经在缓存里了
        current_token = torch.tensor([[next_token]])

    print(f"\n  → 3 步总共计算了 3+1+1 = 5 组 KV")
    print(f"  → 优势：第 2 步开始每次只算 1 组 KV，速度恒定。")
    print(f"  → 生成 N 个 token 的计算量：O(N)")
    print()

    # ================================================================
    # 总结
    # ================================================================
    print("═" * 55)
    print("  核心对比：传给 model 的 vs Attention 看到的")
    print("═" * 55)
    print(f"  {'':12} {'传给 model':>12} {'新算 KV':>10} {'Attention 看到':>15}")
    print(f"  {'─' * 12} {'─' * 12} {'─' * 10} {'─' * 15}")
    print(f"  {'无缓存 第1步':12} {'3 个 token':>12} {'3 组':>10} {'3 个':>15}")
    print(f"  {'无缓存 第2步':12} {'4 个 token':>12} {'4 组':>10} {'4 个':>15}")
    print(f"  {'无缓存 第3步':12} {'5 个 token':>12} {'5 组':>10} {'5 个':>15}")
    print(f"  {'─' * 12} {'─' * 12} {'─' * 10} {'─' * 15}")
    print(f"  {'有缓存 第1步':12} {'3 个 token':>12} {'3 组':>10} {'3 个':>15}")
    print(f"  {'有缓存 第2步':12} {'1 个 token ←':>12} {'1 组 ←':>10} {'4 个':>15}")
    print(f"  {'有缓存 第3步':12} {'1 个 token ←':>12} {'1 组 ←':>10} {'5 个':>15}")
    print()
    print("  结论：Attention 看到的上下文完全一样，")
    print("  但有缓存的版本传入的 token 少了 → 重复计算少了 → 快得多。")


# ============================================================================
# 第 3 部分：单独展示 Q、K、V 的物理来源
# ============================================================================

def show_qkv_origin():
    """
    用一个独立的小例子展示：Q、K、V 到底从哪一行代码产生。

    核心链路：
        token ID (整数)
          → Embedding 查表（变成浮点数向量）
            → 乘 W_Q（变成 Query 向量）
            → 乘 W_K（变成 Key 向量）
            → 乘 W_V（变成 Value 向量）

    三个矩阵 W_Q、W_K、W_V 是独立的，所以同一个 embedding
    向量会变成三种不同的向量——分别用于"提问"、"被匹配"、"提供信息"。
    """
    print("\n" + "═" * 55)
    print("  Q、K、V 从哪里来 —— 逐行代码追踪")
    print("═" * 55)

    # 配置参数（故意设小，方便看输出）
    vocab_size = 10     # 假设词汇表只有 10 个词
    hidden_dim = 8      # 每个词用 8 维向量表示

    # ---- ① 创建 Embedding 表和三个投影矩阵 ----
    # 这四个对象就对应 TinyModel 里的 self.embed, self.W_Q, self.W_K, self.W_V
    embed = nn.Embedding(vocab_size, hidden_dim)   # 10×8 的嵌入表
    w_q = nn.Linear(hidden_dim, hidden_dim, bias=False)  # W_Q: 8×8
    w_k = nn.Linear(hidden_dim, hidden_dim, bias=False)  # W_K: 8×8
    w_v = nn.Linear(hidden_dim, hidden_dim, bias=False)  # W_V: 8×8

    # ---- ② 模拟输入："我爱北京"分词后的 token IDs ----
    # 实际场景：tokenizer.encode("我爱北京") → [1053, 2847, 12648]
    # 这里用简单数字代替
    token_ids = torch.tensor([3, 7, 2])  # 3 个 token
    print(f"\n  ① 分词结果: token IDs = {token_ids.tolist()}")
    print(f"     三个整数，每个代表词汇表中的一个词")

    # ---- ③ Token ID → Embedding ----
    embeddings = embed(token_ids)  # [3, 8] — 3 个 token，每个 8 维
    print(f"\n  ② Embedding 查表: 形状 {list(embeddings.shape)}")
    print(f"     token 3 → embed[3] = {embeddings[0].detach().numpy().round(2)}")
    print(f"     token 7 → embed[7] = {embeddings[1].detach().numpy().round(2)}")
    print(f"     token 2 → embed[2] = {embeddings[2].detach().numpy().round(2)}")

    # ---- ④ Embedding × W_Q / W_K / W_V → Q、K、V ----
    Q = w_q(embeddings)  # [3, 8] — 每个 token 的 Query
    K = w_k(embeddings)  # [3, 8] — 每个 token 的 Key
    V = w_v(embeddings)  # [3, 8] — 每个 token 的 Value

    print(f"\n  ③ Embedding × 三个矩阵 → Q、K、V")
    print(f"     Q = embed @ W_Q^T, 形状 {list(Q.shape)}")
    print(f"     K = embed @ W_K^T, 形状 {list(K.shape)}")
    print(f"     V = embed @ W_V^T, 形状 {list(V.shape)}")

    # ---- ⑤ 展示同一个 token 产生三种不同向量 ----
    # 取第一个 token（ID=3），看它的 embedding 乘三个矩阵后得到的值完全不同
    print(f"\n  ④ 同一个 token 的三个身份（以 token[0] 为例）")
    print(f"     token[0] 的 embedding:  {embeddings[0].detach().numpy().round(2)}")
    print(f"     乘 W_Q → Q[0]:         {Q[0].detach().numpy().round(2)}  ← '我想问什么'")
    print(f"     乘 W_K → K[0]:         {K[0].detach().numpy().round(2)}  ← '我能匹配什么'")
    print(f"     乘 W_V → V[0]:         {V[0].detach().numpy().round(2)}  ← '我的实际内容'")
    print(f"\n     同一个 embedding，三个矩阵 → 三个不同的向量。")
    print(f"     这就像同一个人在不同场合：")
    print(f"       Q = 你作为'提问者'的角色")
    print(f"       K = 你作为'回答者'的角色（供别人匹配）")
    print(f"       V = 你说的具体内容")

    # ---- ⑥ 验证 Q 的第一行是干什么用的 ----
    # Q[0] 会和所有 K 做点积，看看 token[0] 最关注谁
    scores_q0 = Q[0] @ K.T  # Q[0] 和 K[0]、K[1]、K[2] 分别做点积
    print(f"\n  ⑤ Q[0] 和所有 K 的点积分数: {scores_q0.detach().numpy().round(2)}")
    print(f"     分数最高的是 K[{scores_q0.argmax().item()}] — token[0] 最关注这个 token")


# ============================================================================
# 运行入口
# ============================================================================

if __name__ == "__main__":
    # 按这个顺序看：
    # 1. Q、K、V 的物理来源（5 行代码讲清楚）
    show_qkv_origin()

    # 2. 训练做了什么（W_Q、W_K、W_V 怎么来的）
    trained_model = simulate_training()

    # 3. 推理时有无 KV Cache 的区别
    inference_demo(trained_model)
