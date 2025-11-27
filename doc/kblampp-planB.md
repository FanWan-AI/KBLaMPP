# KBLaM++ Plan B：小参数模型下的实验与工程实施细则

## 目标
在 2080Ti / RTX3060 / 单张租用 4090 这类显卡条件下，跑通 KBLaM++ 的全流程 Demo：
从五元组构建 → K/V 编码 → ANN 检索 → KBLaM++ 注入 → 训练 & 评测 → 证据链输出。
模型规模控制在 1B–3B 级别，后续可无缝切到大模型（A800/H100）版本。

## 0. 我们在做什么（Plan B 核心思路）

KBLaM++ 的理论不变，Plan B 只是换成更"轻量"的工程配置：

- 把知识库中的事实统一整理为五元组
$$T_i = (h_i, r_i, t_i, c_i, \tau_i)$$
- 其中：
  - $h_i, r_i, t_i$：实体–关系–实体 / 值；
  - $c_i$：证据句（context.sent_span）；
  - $\tau_i = (\tau_i^{\min}, \tau_i^{\max})$：时间窗。
- 用句向量模型对文本字段编码，经过 MLP 映射得到：
  - Key 向量 $K_i \in \mathbb{R}^{d_k}$
  - Value 向量 $V_i \in \mathbb{R}^{d_v}$
- 把所有 Key 构建成ANN 索引（HNSW / IVF-PQ）。
- 在线推理时，在小 LLM 的某一层，对每个 token 产生查询向量 $Q_j$：
  a. 用 ANN 在 Key 上取出 Top-k 候选；
  b. 对候选做 "语义 + 上下文 + 时间" 综合打分：
$$s_{ij} = \text{sem}_{ij} + \gamma \cdot \phi_{ij} + \eta \cdot g^{\text{time}}_{ij}$$
  c. softmax 得到权重 $\alpha_{ij}$，对 Value 加权平均得到知识注入向量 $\widetilde{V}^{(j)}$；
  d. 通过一条知识注意力分支 + 门控 $\beta_j$ 与原始文本注意力输出融合。

整个链路可解释：我们可以导出"问题–答案–用到的五元组–权重–时间窗"。

## 1. 模型选型（适配小显卡）

### 1.1 Backbone LLM（小参数版本）

目前可用的的备选：
1. meta-llama/Llama-3.2-1B-Instruct
2. Qwen/Qwen2.5-3B-Instruct
3. meta-llama/Llama-3.2-3B-Instruct

这些架构都可以做 KBLaM++ 注入，理由：
- 都是 HuggingFace transformers 标准接口的解码式 Transformer（decoder-only）。
- 都有一堆重复的 TransformerBlock/DecoderLayer，每层输出 hidden states，我们只需要在某几层插入：
  - 一层 linear_q：$h \to Q$
  - 一层 linear_v：$\widetilde{V} \to d_\text{model}$
  - 一层 KBFusionLayer：并行注意力 + 门控
- 不改变原有权重结构，只是加模块 & 前向逻辑包装，架构完全兼容。

**Plan B 推荐策略：**
- **默认 backbone**：Qwen/Qwen2.5-3B-Instruct
  - 3B 体量在 24GB 显卡上，配合 4-bit 量化 + LoRA + seq_len=512，batch size 1–2 仍然可行；
  - 中英能力较好，后面扩到中文任务也方便。
- **超省显存备选**：meta-llama/Llama-3.2-1B-Instruct
  - 在 2080Ti / 3060 上也能较舒适地跑（配合 4-bit 或 8-bit）；
  - 适合先完整打通流程：五元组 → KV → ANN → 注入 → 训练。
- **可选**：meta-llama/Llama-3.2-3B-Instruct
  - 和 Qwen2.5-3B 类似，二选一即可，无强制。

实际实现中，用一个统一 wrapper：
只要 d_model、层数、注入层编号一致，换 backbone 只需要改配置。

```yaml
# configs/backbone_llama1b.yaml
backbone: "meta-llama/Llama-3.2-1B-Instruct"
d_model: 2048   # 以模型 config 为准
```

### 1.2 Sentence Embedding 选型（小显卡友好）

目前可用的备选：
1. sentence-transformers/all-MiniLM-L6-v2
  - 维度：384
  - 优点：极快、资源占用小，非常适合 Plan B。
2. Qwen/Qwen3-Embedding-0.6B
  - 维度：官方是 1024（以 HF 为准）
  - 模型较大，编码速度 & 显存占用比 MiniLM 大。
3. sentence-transformers/paraphrase-multilingual-mpnet-base-v2
  - 维度：768
  - 支持多语言；但模型比 MiniLM 大一些。
4. BAAI/bge-small-en-v1.5
  - 维度：384
  - 高质量英文嵌入，小模型，训练友好。

**Plan B 推荐默认：**
- **首选**：BAAI/bge-small-en-v1.5 或 all-MiniLM-L6-v2
  - 维度 384，小而精，足够应付 T-REx / Hotpot / TimeQA / 合成任务；
  - 适合 CPU/小显卡离线编码。
- 如需多语言，再考虑 paraphrase-multilingual-mpnet-base-v2。

**实现建议：**
- 在 configs/*.yaml 里统一说明：
```yaml
embedding_model: "BAAI/bge-small-en-v1.5"
d_ctx: 384   # 自动从模型 config 检测，或者写死
```
- offline/encode_kv.py 内部通过 HF AutoModel+AutoTokenizer 加载 sentence embedding 模型，使用 model(**inputs).pooler_output 或指定 pooling 策略。

## 2. 仓库结构（小模型版与大模型版完全共用）

目录结构：
```
kblam_pp/
  README.md                    # 本文档（Plan B 实施细则可以放在此）
  configs/
    backbone_llama1b.yaml      # Llama-3.2-1B-Instruct 配置
    backbone_qwen3b.yaml       # Qwen2.5-3B-Instruct 配置
    synth_world.yaml           # 合成世界+QA 配置
    trex.yaml                  # T-REx 实验配置（可做子集）
    hotpot.yaml                # HotpotQA 实验配置（可做子集）
    timeqa.yaml                # TimeQA 实验配置（可做子集）
  data/
    raw/                       # 原始数据 (T-REx/Hotpot/TimeQA + LLM 生成)
    5tuple/                    # 统一五元组 JSONL
    qa/                        # 统一 QA JSONL
  store/
    K.npy                      # (N, d_k)
    V.npy                      # (N, d_v)
    index_hnsw/                # ANN 索引
    meta/
      ctx_vec.npy              # (N, d_ctx)
      tau_min.npy              # (N,)
      tau_max.npy              # (N,)
      entity_ids.npy           # (N,)
      rel_ids.npy              # (N,)
  kblampp/
    __init__.py
    knowledge_index.py         # ANN 封装
    kb_store.py                # K/V/meta 封装
    scorers.py                 # ContextScorer / TimeScorer
    selector.py                # sem+ctx+time -> alpha -> V_tilde
    fusion.py                  # KBFusionLayer
    injection_wrapper.py       # 注入到 Llama/Qwen 的封装
  offline/
    build_5tuple_from_trex.py
    build_5tuple_from_hotpot.py
    build_5tuple_from_timeqa.py
    gen_synth_world.py         # LLM 生成 world (entities + facts)
    gen_synth_qa.py            # 基于 world 生成 QA
    encode_kv.py               # 句向量 + MLP 生成 K/V + meta
    build_index.py             # 构建 ANN 索引
    sanity_check_5tuple.py     # schema + 维度检查
  train/
    train_stageA.py
    train_stageB.py
    dataloader.py
    metrics.py
  eval/
    eval_qa.py
    dump_evidence.py
    visualize_alpha.py
  infer/
    infer_server.py
```

Plan B 和 Plan A 的唯一区别：configs 中的模型规模和训练超参不同。

## 3. 数据结构与维度（Plan B 不变）

五元组 schema 沿用已经定下的格式，这里只补一句：Plan B 完全复用，不再改 schema。

**维度符号（与小模型无关，通用）：**

| 符号 | 含义 | Plan B 建议值 |
|------|------|---------------|
| N | KB 五元组总数 | 10⁵ ~ 10⁶（Plan B 可先 10⁴–10⁵） |
| B | batch size（问题数） | 1–4 |
| T | 序列长度（token 数） | 256 / 512 |
| K | 每 token 的候选知识数 | 8 / 16 |
| d_model | LLM 隐状态维度 | 1B:2048, 3B:3k–4k |
| d_ctx | context 句向量维度 | 384 / 768 / 1024 |
| d_k | Key 维度 | 256 / 384 / 512 |
| d_v | Value 维度 | 256 / 384 / 512 |
| d_τ | 时间编码维度 | 32 / 64 |

**Plan B 的经验配置：**
- d_ctx = 384（MiniLM / bge-small）
- d_k = d_v = 384
- 这样 K/V MLP 都很轻量，适配小模型。

## 4. 离线流程（对小显卡友好地做完所有准备）

整个离线步骤可以全部在 CPU + 小 embedding 模型 上完成，不吃 GPU 训练资源。

### 4.1 Step 1：构建五元组（真实数据 + LLM 合成）

**脚本：**
- offline/build_5tuple_from_trex.py
- offline/build_5tuple_from_hotpot.py
- offline/build_5tuple_from_timeqa.py
- offline/gen_synth_world.py
- offline/gen_synth_qa.py

**Plan B 建议：**
- 第一步只用"合成 world + QA" 做一个小 KB（比如几千条五元组）：
  - **好处：**
    - 完全可控；
    - 多跳结构可以设计得很清晰；
    - KB 规模小 → ANN & 训练都更轻量。
- 后面再逐步加 T-REx / Hotpot / TimeQA 的子集。

#### 4.1.1 合成 world 五元组 Prompt（英文，多跳友好）

offline/gen_synth_world.py 中使用的 GPT-4 / Qwen-Plus Prompt（示例）：
```
You are a data generator for a knowledge-augmented language model.

Task:
1. Create a small fictional world with:
   - 10-15 entities (people, organizations, locations, events),
   - 60-80 facts connecting them.

2. Each fact must be expressed as a 5-tuple in JSON with the following schema:
   {
     "head":    { "id": "E001", "name": "BrightSky Robotics", "type": "ORG" },
     "relation":{ "id": "R001", "name": "chief executive officer" },
     "tail":    { "id": "E002", "name": "Alice Zhang", "type": "PERSON" },
     "context": {
       "source": "llm_synth",
       "page_title": "BrightSky Robotics",
       "sent_span": "Alice Zhang became the CEO of BrightSky Robotics on March 2, 2019.",
       "disamb": { "country": "US" },
       "ver": { "url": null, "rev_id": null }
     },
     "time_window": {
       "start": "2019-03-02",
       "end": null,
       "source": "text_infer"
     }
   }

Requirements:
- Use short string IDs for head.id and tail.id (like "E001", "E002", ...).
- Use short string IDs for relation.id (like "R001", "R002", ...).
- Ensure that:
  - There are one-hop facts,
  - There are 2-hop and 3-hop chains (for multi-hop reasoning),
  - Some facts are time-dependent (e.g., positions, memberships with start/end dates).

Output:
- A pure JSON array of 5-tuples (no extra commentary).
```

这样 world 里就自然有多跳链（2-hop, 3-hop），并且每条事实都已经是我们的五元组格式。

#### 4.1.2 合成 QA Prompt（单跳+多跳+时态）

offline/gen_synth_qa.py 使用 world JSON 作为输入：
```
You are given a fictional world described by a list of facts in 5-tuple format.
Each fact T_i has fields: head, relation, tail, context.sent_span, time_window.

Your tasks:
1. Generate SINGLE-HOP questions that can be answered using exactly ONE fact.
2. Generate MULTI-HOP questions that require combining TWO or THREE facts.
3. Generate TEMPORAL questions that depend on the time_window (start/end).

For each question, output a JSON object:
{
  "qid": "Q0001",
  "dataset": "synth_world_01",
  "question": "...",
  "answer": "...",
  "type": "single-hop" | "multi-hop" | "temporal",
  "supporting_facts": [indices_of_used_facts],
  "question_time": { "start": "...", "end": "..." }
}

Notes:
- supporting_facts should list the indices of the facts in the original world array (0-based).
- For temporal questions, question_time should reflect the time constraint in the question.
- Output a pure JSON array of such question objects.
```

如此我们可以拿到：
- 有显式 supporting_facts（方便监督 $\alpha$）
- 有 question_time（时态模块用）

#### 4.1.3 T-REx / Hotpot / TimeQA 子集（Plan B 可选）

在显存有限时，建议：
- T-REx：随机采样 1–5 万条 (h, r, t)，构建对应五元组；
- Hotpot：随机采样 5k–10k QA 对，抽取 supporting facts；
- TimeQA：随机采样 5k 左右。

可以在 build_5tuple_from_*.py 内加 `--max_samples` 选项。

### 4.2 Step 2：句向量 + K/V 编码（CPU 友好）

**脚本：** offline/encode_kv.py

**Plan B 建议：** 全部在 CPU 上跑，不占 GPU 显存。

**流程：**
1. 加载 embedding_model（如 BAAI/bge-small-en-v1.5）。
2. 遍历所有五元组 JSONL：
  - 对 head.name, relation.name, tail.name, context.sent_span 分别编码：
    - 得到向量 $e^h_i, e^r_i, e^t_i, e^c_i \in \mathbb{R}^{d_\text{ctx}}$
  - 把 time_window.start/end 转成相对天数，输入一个小 MLP 得到时间向量 $e^\tau_i \in \mathbb{R}^{d_\tau}$。
3. 用轻量 MLP 映射到 K/V：
$$K_i = \mathrm{MLP}_k(e^h_i \oplus e^r_i) \in \mathbb{R}^{d_k}$$
$$V_i = \mathrm{MLP}_v(e^t_i \oplus e^c_i \oplus e^\tau_i) \in \mathbb{R}^{d_v}$$
4. 保存到 store/：
  - K.npy：$(N, d_k)$
  - V.npy：$(N, d_v)$
  - meta/ctx_vec.npy：所有 $e^c_i$
  - meta/tau_min.npy、meta/tau_max.npy
  - meta/entity_ids.npy、meta/rel_ids.npy

### 4.3 Step 3：构建 ANN 索引（faiss-cpu 即可）

**脚本：** offline/build_index.py，不需要 GPU。
- 使用 faiss.IndexHNSWFlat 或 IVF-PQ；
- 数据量不大时（N ≤ 10⁵），HNSW 即可。
- 训练命令：
```bash
python offline/build_index.py \
  --store_dir store/ \
  --method hnsw \
  --similarity cosine
```

## 5. 在线 KBLaM++ 架构（对小模型的插入方式）

架构逻辑与大模型一样，只是注入层数少、维度更小：

1. 在 backbone 的 config 里读取 num_hidden_layers 和 hidden_size (d_model)。
2. 选择 1–2 个注入层（例如：1B 模型在第 8 层，3B 在第 16、24 层）。
3. 在这些层上加上：
  - linear_q: d_model -> d_k
  - KBSelector（使用 K/V + ANN）
  - linear_v: d_v -> d_model
  - KBFusionLayer: (h, V_tilde_proj) -> Y
4. 其余层保持不变。

**小模型特化建议：**
- 对 1B 模型：
  - inject_layers: [8] 足够；
  - d_k = d_v = 256 / 384；
  - K_top = 8。
- 对 3B 模型：
  - inject_layers: [12, 20]（中層 + 高层）；
  - d_k = d_v = 384 / 512；
  - K_top = 16。

## 6. 训练流程（专门为 1B–3B 显卡预算设计）

### 6.1 硬件预算与建议

- **2080Ti (11GB)：**
  - 推荐：Llama-3.2-1B-Instruct，seq_len=512，batch_size=1；
  - 使用 4-bit/8-bit 量化 + LoRA（比如 bitsandbytes + peft）；
  - 只跑 Stage A 也足够展示 KBLaM++ 效果。
- **RTX 3060 (12GB)：**
  - 类似 2080Ti 配置，或者更 aggressive：batch_size=2。
- **租用单张 4090 (24GB)：**
  - 可使用 Qwen2.5-3B-Instruct，seq_len=512，batch_size=2–4；
  - Stage B（微调顶层/LoRA）也可尝试。

### 6.2 阶段 A：冻结 backbone，只训 KB 模块

**目标：**
让选择器 $\alpha_{ij}$ 和门控 $\beta_j$ 学会合理使用知识，在 QA 任务上有提升。

**需要训练的参数：**
- linear_q、linear_v
- KBSelector（ContextScorer + TimeScorer 内部参数）
- KBFusionLayer（text_mha, kb_mha 可与 backbone MHA 共享 or 单独，Plan B 建议单独但维度一样）
- 可选：最后一两层 LoRA（如果显存允许）

**不训练：**
- backbone LLM 的基础权重（全部 requires_grad=False）。

**Config 示例：** configs/synth_world_stageA_llama1b.yaml
```yaml
backbone: "meta-llama/Llama-3.2-1B-Instruct"
embedding_model: "BAAI/bge-small-en-v1.5"

d_model: 2048
d_k: 384
d_v: 384
d_ctx: 384

inject_layers: [8]
K_top: 8
gamma: 1.0
eta: 1.0
temperature: 1.0

train_dataset: "data/qa/synth_world_train.jsonl"
eval_dataset:  "data/qa/synth_world_dev.jsonl"

optimizer:
  type: "adamw"
  lr: 5e-4      # 只训 KB 模块，可以稍大
  weight_decay: 0.01

batch_size: 1
max_steps: 5000
grad_accum: 4      # 等效 batch_size=4
max_seq_len: 512
log_interval: 50
save_interval: 1000
```

**训练脚本大致流程（train/train_stageA.py）：**

1. 加载 backbone + tokenizer
2. 冻结 backbone 参数  
3. 初始化 KB 模块
4. 加载 KnowledgeIndex 和 KBValueStore
5. 循环读取 QA 样本，按以下流程前向：
   - 编码问题 → hidden states 至注入层 L；
   - 用 linear_q 得到查询向量 Q；
   - 展平成 (B×T, d_k) 维度，ANN 检索 → 候选索引 I；
   - KBValueStore.fetch(I) → 知识库的 Key、Value、上下文向量、时间窗最小值、时间窗最大值等元数据；
   - KBSelector(Q, ...) → 注意力权重 α 和加权后的知识向量 V_tilde；
   - linear_v(V_tilde) → 投影到模型维度的知识向量 V_kb_tilde；
   - KBFusionLayer(h_L, V_kb_tilde) → 新的隐藏状态 h_L'；
   - 继续后续 Transformer 层；
   - 调用 LM head 生成/分类答案，计算交叉熵损失 CE loss。
6. optimizer.step() 只更新 KB 模块。

### 6.3 阶段 B：解冻顶层 / LoRA（可选）

在显存允许时（推荐在 4090 上）：
- 解除最后 2–4 层的 requires_grad 或加载 LoRA 适配器；
- 学习率降到 1e-5 量级；
- 可以加辅助损失（如果我们有 supporting_facts）：
  - 对路径内的五元组，鼓励它们的 $\alpha_{ij}$ 高；
  - 对时间严重不匹配的事实，惩罚高权重。

Plan B 中，这一步可以先不做，把焦点放在跑通 Stage A + 看得见证据链。

## 7. 评测与可解释性（小规模实验版）

### 7.1 快速 sanity check 流程

在 1B backbone + 合成 world 的配置下，可以先跑一套tiny 实验：
1. 合成 KB world：200–500 条五元组；
2. 合成 QA：500–2000 题；
3. 训练 Stage A：几千步就够（15–60 分钟级别，视设备而定）；
4. eval/eval_qa.py 算单跳 / 多跳 / 时态问答的 EM/F1；
5. eval/dump_evidence.py 导出若干样例的证据链 JSON；
6. eval/visualize_alpha.py 画出权重条形图。

**只要能看见：**
- 模型答案正确率明显高于"只用 backbone 不用 KB"；
- 正确答案对应的事实在 Top-k 中有较高 $\alpha$；
- 时间窗匹配的事实权重更大；

就说明整个 KBLaM++ 流程在小模型上跑通了。

### 7.2 证据链输出格式（不区分 Plan A/B）

eval/dump_evidence.py 输出类似：
```json
{
  "qid": "Q0001",
  "dataset": "synth_world_01",
  "question": "Who is the CEO of BrightSky Robotics in 2019?",
  "answer": "Alice Zhang",
  "pred_answer": "Alice Zhang",
  "top_k_facts": [
    {
      "index": 123,
      "head": "BrightSky Robotics",
      "relation": "chief executive officer",
      "tail": "Alice Zhang",
      "context": "Alice Zhang became the CEO of BrightSky Robotics on March 2, 2019.",
      "time_window": ["2019-03-02", null],
      "alpha": 0.82
    },
    {
      "index": 57,
      "head": "BrightSky Robotics",
      "relation": "chief executive officer",
      "tail": "David Kim",
      "context": "...",
      "time_window": ["2015-01-01", "2018-12-31"],
      "alpha": 0.12
    }
  ]
}
```

## 8. 针对小显卡的几条实践建议（工程侧注意事项）

1. **强烈建议先只用合成 world 做 Demo：**
  - KB 规模、QA 数量都可控；
  - 显存 & 训练时间压力最小；
  - 能完整体现单跳、多跳、时态三种能力。

2. **全面使用 4-bit/8-bit + LoRA：**
  - backbone 用 bitsandbytes 量化；
  - KB 模块保持 fp16/fp32 即可，不大；
  - 这样 3B 模型在单张 4090 上也能训练。

3. **先做最小配置，再做完整配置：**
  - **最小 Demo：**
    - backbone: Llama-3.2-1B-Instruct
    - d_k = d_v = 256
    - K_top = 8
    - 合成 KB：~1000 五元组
    - QA：~1000 条
  - **完整 Demo：**
    - backbone: Qwen2.5-3B-Instruct
    - d_k = d_v = 384/512
    - 合成 KB + T-REx 子集 + TimeQA 子集

4. **所有超参写进 configs/*.yaml，不在代码硬编码：**
  - 包括：backbone、embedding_model、d_k/d_v/d_ctx、inject_layers、K_top、gamma/eta/temperature。

5. **任何对公式的改动一律禁止：**
  - 核心计算必须保持：

$$s_{ij} = \text{sem}_{ij} + \gamma\phi_{ij} + \eta g^{\text{time}}_{ij}$$

$$\alpha = \mathrm{softmax}(s)$$

$$\widetilde{V}^{(j)} = \sum_i \alpha_{ij} V_i$$

## 9. 总结（给第三方工程实现团队看的"一句话 checklist"）

只要按下面三个层次一步步往下做，就能在 1B–3B 模型上跑通 KBLaM++：

1. **离线准备：**
  - 用 LLM 生成一个小型世界（entities + 5-tuple facts）；
  - 根据 world 生成 QA（带 supporting_facts + question_time）；
  - 用 bge-small-en / MiniLM 对五元组编码，生成 K/V + meta；
  - 用 faiss 构建 ANN 索引。

2. **在线结构：**
  - 在 Llama-3.2-1B 或 Qwen2.5-3B 的几层之间插入 KBLaM++ 模块：
    - linear_q：hidden → Q；
    - KBSelector：ANN 检索 + sem + ctx + time 打分；
    - linear_v：V_tilde → d_model；
    - KBFusionLayer：文本注意力输出 + 知识分支输出 + 门控 $\beta$。

3. **训练与评测：**
  - Stage A：冻结 backbone，只训练 KB 模块，任务是 QA；
  - 在合成 QA 上得到合理的 EM/F1；
  - 导出问题–答案–证据链 JSON，肉眼看一看 Alpha 是否"用对了知识"。

满足以上三点，即使是在 1B–3B 小模型上，也已经是一个完整、可展示、可解释的 KBLaM++ Plan B 实现。