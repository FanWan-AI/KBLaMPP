# KBLaM++ 技术实现文档

## 0. 总览：我们在做什么？

KBLaM++ 是对 KBLaM 的增强版，核心思想：

- 把知识库中的事实统一整理为五元组
$$T_i = (h_i, r_i, t_i, c_i, \tau_i)$$

其中：
  - $h_i, r_i, t_i$：实体-关系-实体/值；
  - $c_i$：证据句；
  - $\tau_i = (\tau_i{\min}, \tau_i{\max})$：时间窗；

- 对每条五元组编码出一个 Key 向量 $K_i$ 和一个 Value 向量 $V_i$；
- 将所有 Key 放入 ANN 索引（HNSW / IVF-PQ）；
- 在线推理时，大模型在某一层对每个 token 产生查询向量 (Q_j)：
  - 用 ANN 在 Key 上取出 Top-k 候选；
  - 对候选做"语义 + 上下文 + 时间"综合打分 $s_{ij}$，softmax 得到权重 $\alpha_{ij}$；
  - 对 Value 做加权平均得到知识注入向量 $\widetilde{V}^{(j)}$；
  - 通过一条知识注意力分支 + 门控 $\beta_j$ 与原始文本注意力输出融合。

整个链路保持可解释：我们始终知道用了哪条五元组，权重是多少，时间窗是否匹配。

## 1. 模型与依赖（与 KBLaM 对齐）

### 1.1 Backbone LLM

- **模型**： Llama 3 8B Instruct
  - 示例（HuggingFace）：`meta-llama/Meta-Llama-3-8B-Instruct`
- **使用方式**：
  - 阶段 A：冻结全部 Llama 权重，只训练 KBLaM++ 模块（KV 编码、选择器、门控融合）。
  - 阶段 B：解冻顶层若干层或使用 LoRA 进行联合微调。

### 1.2 句向量 / 基础 encoder

- **模型**： OpenAI text-embedding-ada-002
- **维度**： $P = 1536$
- **用途**：
  - 对以下文本字段编码：
    - head.name
    - relation.name  
    - tail.name
    - context.sent_span
  - 得到基础向量后，再通过 MLP 映射到$(K_i, V_i)$。

和 KBLaM 一致：
> "For all experiments, we use the instruction fine-tuned version of Llama3 8B … and OpenAI's ada-002 sentence embedding model (P = 1536) as the pre-trained encoder for computing base key and value embedding."

### 1.3 环境依赖

建议环境（示例）：
- Python ≥ 3.10
- PyTorch ≥ 2.1 + CUDA（推荐 A100 / 4090 / H100 等）
- 依赖包（简略）：
  - transformers
  - accelerate
  - faiss-gpu 或 faiss-cpu
  - numpy, scipy, pandas
  - openai（或兼容客户端，用于调用 ada-002 和 GPT-4 系列）
  - pyyaml, tqdm, datasets

## 2. 仓库结构约定

建议采用如下目录结构（可以微调，但语义不要变）：

```
kblam_pp/
  README.md                    # 本文档
  configs/
    synth_world.yaml           # 合成世界+QA 配置
    trex.yaml                  # T-REx 实验配置
    hotpot.yaml                # HotpotQA 实验配置
    timeqa.yaml                # TimeQA 实验配置
  data/
    raw/                       # 原始数据 (T-REx/Hotpot/TimeQA + LLM 生成的 JSON)
    5tuple/                    # 统一五元组 JSONL (每行一个 five-tuple)
    qa/                        # 统一 QA JSONL (问题-答案-支持路径)
  store/
    K.npy                      # 所有 K_i (N, d_k)
    V.npy                      # 所有 V_i (N, d_v)
    index_hnsw/                # ANN 索引文件
    meta/
      ctx_vec.npy              # 句向量 (N, d_ctx = 1536)
      tau_min.npy              # (N,)
      tau_max.npy              # (N,)
      entity_ids.npy           # (N,) head/tail 标识
      rel_ids.npy              # (N,) relation 标识
      # 其他需要的 meta
  kblampp/
    __init__.py
    knowledge_index.py         # ANN 封装
    kb_store.py                # K/V/meta 取值封装
    scorers.py                 # ContextScorer / TimeScorer
    selector.py                # sem + ctx + time -> alpha -> V_tilde
    fusion.py                  # KBFusionLayer 并行注意力+门控
    injection_wrapper.py       # 将 KBLaM++ 注入 Llama3 的封装
  offline/
    build_5tuple_from_trex.py  # T-REx -> 五元组
    build_5tuple_from_hotpot.py
    build_5tuple_from_timeqa.py
    gen_synth_world.py         # 调 LLM 生成合成世界（entities + facts）
    gen_synth_qa.py            # 基于 world 生成 QA
    encode_kv.py               # 调 ada-002 + MLP 生成 K/V + meta
    build_index.py             # 构建 ANN 索引
    sanity_check_5tuple.py     # schema、维度检查脚本
  train/
    train_stageA.py            # 冻结 Llama，只训 KB 模块
    train_stageB.py            # 解冻顶层 / LoRA 联合训练
    dataloader.py              # 统一加载 QA + question_time + supporting_facts
    metrics.py                 # EM / F1 / 时间指标等
  eval/
    eval_qa.py                 # 评估脚本（支持多数据集）
    dump_evidence.py           # 导出问题-答案-Top-k-五元组-权重 JSON
    visualize_alpha.py         # 画条形图 / 热力图等
  infer/
    infer_server.py            # Online Demo 服务（REST / CLI）
```

## 3. 核心数据结构：五元组与维度说明

### 3.1 五元组 JSON Schema

存储格式： `data/5tuple/*.jsonl`，每行一个 JSON，格式如下：

```json
{
  "head": {
    "id": "Q68", 
    "name": "Microsoft", 
    "type": "ORG"
  },
  "relation": {
    "id": "P169", 
    "name": "chief executive officer"
  },
  "tail": {
    "id": "Q180266", 
    "name": "Satya Nadella", 
    "type": "PERSON"
  },
  "context": {
    "source": "wikipedia.en",
    "page_title": "Satya Nadella",
    "sent_span": "Satya Nadella became the CEO of Microsoft on February 4, 2014.",
    "disamb": {
      "org_alias": ["Microsoft", "MSFT"],
      "country": "US"
    },
    "ver": {
      "url": "...",
      "rev_id": 12345678
    }
  },
  "time_window": {
    "start": "2014-02-04",
    "end": null,
    "source": "P580"
  }
}
```

合成数据将 `context.source` 置为 `"llm_synth"`，`time_window.source` 为 `"text_infer"` 即可。

### 3.2 维度符号说明表

| 符号 | 含义 | 示例值（建议） |
|------|------|----------------|
| N | KB 中五元组的总数 | 10^5 ~ 10^7 |
| B | batch size（问题个数） | 4 ~ 32 |
| T | 每个问题的 token 数 | 128 ~ 512 |
| K | 每个 token ANN 返回的候选数 | 8 / 16 / 32 |
| d_model | Llama3 隐状态维度 | 4096（以实际 config 为准） |
| P | ada-002 句向量维度 | 1536 |
| d_ctx | context 句向量内部使用维度 | 一般 = 1536 |
| d_k | Key 维度 | 512 / 768 / 1024 |
| d_v | Value 维度 | 512 / 768 / 1024 |
| d_τ | 时间编码维度 | 32 / 64 |

**典型数据张量的形状：**

| 名称 | 形状 | 含义说明 |
|------|------|-----------|
| K.npy | (N, d_k) | 所有五元组的 Key 向量集合 |
| V.npy | (N, d_v) | 所有五元组的 Value 向量集合 |
| ctx_vec.npy | (N, d_ctx) | context.sent_span 的句向量 |
| tau_min.npy | (N,) | 每条五元组时间窗起点（数值形式，例如天数） |
| tau_max.npy | (N,) | 每条五元组时间窗终点 |
| Q | (B, T, d_k) | 某一注入层上，所有 token 的查询向量 |
| K_kb | (B, T, K, d_k) | 对每个 token，从 ANN 取回的 K_i |
| V_kb | (B, T, K, d_v) | 对每个 token，对应的 Value 向量 |
| ctx_vec_batch | (B, T, K, d_ctx) | 对应候选五元组的 context 句向量 |
| rel_id_batch | (B, T, K) | 对应候选的 relation 标识（int 索引） |
| ent_id_batch | (B, T, K) | 对应候选的实体类型/ID 索引 |
| tau_min_batch | (B, T, K) | 候选事实的时间窗起点 |
| tau_max_batch | (B, T, K) | 候选事实的时间窗终点 |
| alpha | (B, T, K) | 对每个 token/候选的权重 |
| V_tilde | (B, T, d_v) | 知识注入向量 |
| V_tilde_proj | (B, T, d_model) | 投射到 Llama 维度后的知识向量 |
| Y_txt, Y_kb, Y | (B, T, d_model) | 文本注意力输出 / 知识分支输出 / 融合后输出 |

## 4. 离线流程：五元组构建与 KV 编码

### 4.1 第一步：从真实数据构建五元组

**脚本入口：**
- `offline/build_5tuple_from_trex.py`
- `offline/build_5tuple_from_hotpot.py`  
- `offline/build_5tuple_from_timeqa.py`

**统一输出：**
- `data/5tuple/trex.jsonl`
- `data/5tuple/hotpot.jsonl`
- `data/5tuple/timeqa.jsonl`

**核心原则（简要）：**

1. **T-REx**：
   - 每个 (h, r, t) + 对齐的 Wikipedia 句 → 一条五元组；
   - time_window 从 Wikidata 中 P580/P582/P585 抽取；
   - context.sent_span 为原始证据句。

2. **HotpotQA（多跳）**：
   - 使用官方的 supporting facts（文档 + 句号位置）；
   - 对每条 supporting sentence 中的显式事实抽 (h, r, t)；
   - 支持同一实体/关系多条事实形成局部链。

3. **TimeQA（时态）**：
   - 问题中有显式时间或从证据中抽出时间；
   - 生成时：
     - 五元组的时间窗来自"事实真实发生时间"；
     - QA 样本中的 question_time 来自题面或题中约束（用于 q_min, q_max）。

你可以在脚本中把"抽取规则"写成注释，便于后期替换为更强的 IE 模型。

### 4.2 第二步：用 LLM 合成 KB 与 QA（合成数据）

**脚本入口：**
- `offline/gen_synth_world.py`：
  - 调 GPT-4 生成一个 world：entities + facts；
- `offline/gen_synth_qa.py`：
  - 基于 world 生成单跳、多跳、时态问题。

**Prompt 模板（概述）：**
- **单跳五元组 Prompt**：
  - 生成 50 条独立事实，每条输出为我们定义的 5-tuple JSON；
  - source = "llm_synth"；
- **多跳 world Prompt**：
  - 生成若干实体 + 60~80 条 facts，形成多跳可达图；
- **QA Prompt**：
  - 读入 world JSON，输出 20~40 个 QA 对象，每条包含：
    - question, answer, type, supporting_facts, reasoning_path。

**工程建议：**
- 每次调用 GPT-4 / Qwen Plus生成一个 world，避免上下文过大；
- 所有生成结果保存到 `data/raw/synth_world_*.json` 和 `data/raw/synth_qa_*.json`；
- 用 `offline/sanity_check_5tuple.py` 检查 schema 是否合规。

#### 4.2.1 单跳五元组合成 Prompt（英文）

```
You are a data generator for a knowledge-augmented language model.

Your goal is to create a set of INDEPENDENT factual 5-tuples, each describing
one simple relation between a head entity and a tail entity, possibly with a time window.

For each fact, you MUST output a JSON object with the following fields:

{
  "head":    { "id": "...", "name": "...", "type": "..." },
  "relation":{ "id": "...", "name": "..." },
  "tail":    { "id": "...", "name": "...", "type": "..." },
  "context": {
    "source": "llm_synth",
    "page_title": "...",
    "sent_span": "...",
    "disamb": { ... },
    "ver": { "url": null, "rev_id": null }
  },
  "time_window": {
    "start": "YYYY-MM-DD or null",
    "end":   "YYYY-MM-DD or null",
    "source": "text_infer"
  }
}

Rules:

1. Use SHORT, realistic names, but they MUST be FICTIONAL (no real persons or companies).
2. Use simple relations such as:
   - "chief executive officer", "founded by", "headquarters location",
   - "parent company", "spouse", "born in", "located in", etc.
3. For IDs, use synthetic identifiers:
   - head.id / tail.id: "E_0001", "E_0002", ...
   - relation.id: "R_0001", "R_0002", ...
4. The "sent_span" MUST be a natural English sentence that explicitly expresses (head, relation, tail)
   and, if applicable, a concrete date that can be used as the time_window.start.
   Example: "Alice Zhang became the CEO of BrightSky Robotics on March 2, 2019."
5. If a fact has a clear start date but no known end date, set "end": null.
6. Use "type" for a coarse-grained semantic type: "PERSON", "ORG", "LOC", "EVENT", "PRODUCT", etc.
7. Set "page_title" to a natural title where such a sentence would appear (e.g., the head entity).
8. Set "disamb" to a small JSON object with optional aliases or country codes, or {} if unnecessary.

Finally, output a JSON array of 50 such 5-tuples.
Do NOT include any additional commentary, only valid JSON.
```

#### 4.2.2 多跳局部图 + 五元组合成 Prompt（英文）

```
You are generating a SMALL synthetic knowledge graph for multi-hop reasoning.

Step 1: Create about 20-30 fictional entities of different types:
- companies, people, cities, research labs, products, events, etc.

Step 2: Define a set of relations, for example:
- "founded by", "based in", "subsidiary of",
- "works at", "studied at", "headquartered in",
- "organizes", "takes place in", "acquires", "parent company of".

Step 3: Create around 60-80 factual 5-tuples, each with the following schema:

{
  "head":    { "id": "E_xxxx", "name": "...", "type": "..." },
  "relation":{ "id": "R_yyyy", "name": "..." },
  "tail":    { "id": "E_zzzz", "name": "...", "type": "..." },
  "context": {
    "source": "llm_synth",
    "page_title": "...",
    "sent_span": "...",
    "disamb": { ... },
    "ver": { "url": null, "rev_id": null }
  },
  "time_window": {
    "start": "YYYY-MM-DD or null",
    "end":   "YYYY-MM-DD or null",
    "source": "text_infer"
  }
}

Rules:

1. Entities:
   - Every entity MUST have a unique ID "E_0001", "E_0002", ...
   - Reuse the same entity objects across multiple facts to form a connected graph.

2. Relations:
   - Every relation MUST have a unique ID "R_0001", "R_0002", ...
   - Reuse relation IDs consistently when the semantic relation is the same.

3. Time:
   - Whenever it makes sense (e.g., employment, CEO, acquisition, events),
     include a concrete start date in the sentence and map it to time_window.start.
   - If the relation clearly has an end date, also include it in the sentence
     and map it to time_window.end.
   - Otherwise, set end = null.

4. Context sentence ("sent_span"):
   - MUST explicitly express (head, relation, tail) and the time information
     so that a human can reconstruct the fact.
   - Example: "In June 2018, NovaCloud Inc. acquired the robotics startup Luminary Labs."

5. Graph structure:
   - Make sure there are multi-hop chains of length 2-3.
     For example:
       Entity A works at Company B,
       Company B is a subsidiary of Group C,
       Group C is headquartered in City D.
     This allows questions like:
       "In which city is the company that employs A headquartered?"

Output format:

- First, output a JSON field "entities": a list of all entity objects with id, name, type.
- Second, output a JSON field "facts": a list of all the 5-tuples as described above.

The final output MUST be a single JSON object:
{
  "entities": [...],
  "facts": [...]
}

No extra commentary, only valid JSON.
```

#### 4.2.3 基于合成 KB 的 QA 生成 Prompt（英文，多跳）

```
You are given a small synthetic knowledge graph in JSON format.

The JSON has two fields:
- "entities": a list of entities with (id, name, type).
- "facts": a list of 5-tuples with the schema:

  {
    "head":    { "id": "...", "name": "...", "type": "..." },
    "relation":{ "id": "...", "name": "..." },
    "tail":    { "id": "...", "name": "...", "type": "..." },
    "context": {
      "source": "llm_synth",
      "page_title": "...",
      "sent_span": "...",
      "disamb": { ... },
      "ver": { "url": null, "rev_id": null }
    },
    "time_window": {
      "start": "YYYY-MM-DD or null",
      "end":   "YYYY-MM-DD or null",
      "source": "text_infer"
    }
  }

Your task is to generate QUESTION–ANSWER pairs which can be answered **only** using these facts.

You must produce three types of questions:

1) Single-hop questions (1 fact is enough to answer).
2) Multi-hop questions (2–3 facts are required, e.g., via a chain or a tree).
3) Temporal questions (the answer depends on a time condition:
   "in YEAR", "between YEAR1 and YEAR2", "after YEAR", etc.)

For each question, output a JSON object:

{
  "qid": "Q0001",
  "question": "...",
  "answer": "...",
  "type": "single-hop | multi-hop | temporal",
  "supporting_facts": [ fact_indices ],
  "reasoning_path": "Natural language explanation step by step"
}

Rules:

- "supporting_facts" is a list of integer indices referring to entries in the "facts" array.
- The "answer" must be a short span (an entity name, a date, or a short phrase).
- The "reasoning_path" should explicitly mention which facts are used and how.
- Make sure that:
  - Some questions involve 2-hop reasoning (e.g., A -> B -> C),
  - Some questions involve 3-hop reasoning when possible,
  - Temporal questions use the time_window information correctly.

Output a JSON object:
{
  "qa_pairs": [ ... 20~40 question objects ... ]
}

No extra commentary, only valid JSON.
```

最终，将真实数据和合成数据统一转换为：
- **五元组**：`data/5tuple/*.jsonl`
- **QA**：`data/qa/*.jsonl`

每条 QA 至少包含：

```json
{
  "qid": "Q0001",
  "dataset": "synth_world_01",
  "question": "...",
  "answer": "...",
  "type": "single-hop | multi-hop | temporal",
  "supporting_facts": [3, 5, 10],
  "question_time": {
    "start": "2018-01-01",
    "end": "2018-12-31"
  }
}
```

### 4.3 第三步：KV 编码（ada-002 + MLP）

**脚本**：`offline/encode_kv.py`，主要步骤：

1. 读取所有 `data/5tuple/*.jsonl`，合并到内存或分批处理；
2. 对每条五元组调用 ada-002（注意做缓存，避免重复请求）：
   - head.name → $e^h_i$
   - relation.name → $e^r_i$
   - tail.name → $e^t_i$
   - context.sent_span → $e^c_i$
3. 解析 time_window.start/end 为相对天数（int），输入到一个小 MLP 得到 $e^\tau_i$；
4. 构造：
   - $K_i = \mathrm{MLP}_k(eh_i \oplus er_i)$
   - $V_i = \mathrm{MLP}_v(et_i \oplus ec_i \oplus e^\tau_i)$
5. 保存：
   - `store/K.npy` shape = $(N, d_k)$
   - `store/V.npy` shape = $(N, d_v)$
   - `store/meta/ctx_vec.npy` = 所有 $e^c_i$
   - `store/meta/tau_min.npy`, `tau_max.npy`
   - `store/meta/entity_ids.npy`, `rel_ids.npy`

## 5. ANN 索引与 KBStore 封装

### 5.1 构建 ANN 索引

**脚本**：`offline/build_index.py`

```python
import numpy as np
from kblampp.knowledge_index import KnowledgeIndex

K = np.load("store/K.npy").astype("float32")  # (N, d_k)
index = KnowledgeIndex(
    dim=K.shape[1],
    method="hnsw",      # 或 "ivfpq"
    similarity="cosine" # 内部会做 L2 归一化
)
index.fit(K)
index.save("store/index_hnsw")
```

### 5.2 KnowledgeIndex 接口约定（示例）

`kblampp/knowledge_index.py`：

```python
class KnowledgeIndex:
    def __init__(self, dim, method="hnsw", similarity="cosine"):
        # 初始化 FAISS 索引等

    def fit(self, K: np.ndarray):
        # 构建索引

    def save(self, path: str):
        # 保存索引到目录

    @classmethod
    def load(cls, path: str, gpu: bool = False):
        # 从目录加载索引
        ...

    def query(self, Q: np.ndarray, k: int):
        """
        Q: (M, dim) float32
        返回:
          D: (M, k) 相似度
          I: (M, k) int64 索引 (0..N-1)
        """
```

在线推理时：

```python
index = KnowledgeIndex.load("store/index_hnsw", gpu=True)
D, I = index.query(Q_flat, k=K_TOP)
```

### 5.3 KBValueStore 封装

`kblampp/kb_store.py`：

```python
class KBValueStore:
    def __init__(self, root: str):
        self.K = np.load(f"{root}/K.npy", mmap_mode="r")         # (N, d_k)
        self.V = np.load(f"{root}/V.npy", mmap_mode="r")         # (N, d_v)
        self.ctx = np.load(f"{root}/meta/ctx_vec.npy", mmap_mode="r")
        self.tau_min = np.load(f"{root}/meta/tau_min.npy", mmap_mode="r")
        self.tau_max = np.load(f"{root}/meta/tau_max.npy", mmap_mode="r")
        self.rel_ids = np.load(f"{root}/meta/rel_ids.npy", mmap_mode="r")
        self.ent_ids = np.load(f"{root}/meta/entity_ids.npy", mmap_mode="r")

    def fetch(self, idx: np.ndarray):
        """
        idx: (B*T, K) int64，全局索引
        返回 PyTorch 张量 (已转为 torch.float32 / torch.long)：
          K_kb:      (B, T, K, d_k)
          V_kb:      (B, T, K, d_v)
          ctx_vec:   (B, T, K, d_ctx)
          rel_id:    (B, T, K)
          ent_id:    (B, T, K)
          tau_min:   (B, T, K)
          tau_max:   (B, T, K)
        """
```

## 6. KBLaM++ 核心模块实现要点

### 6.1 ContextScorer 与 TimeScorer

`kblampp/scorers.py` 中实现两个模块：
- **ContextScorer**：语义+上下文匹配；
- **TimeScorer**：时间窗匹配。

你可以直接参考此前给出的示例代码（einsum、bmm 部分记得加中文注释）：
- `ContextScorer.forward(Q, ctx_vec, rel_id, ent_id)` 返回 `[B, T, K]`；
- `TimeScorer.forward(q_min, q_max, tau_min, tau_max)` 返回 `[B, T, K]`。

### 6.2 Selector：综合打分 + 归一化 + V_tilde

`kblampp/selector.py`：

```python
class KBSelector(nn.Module):
    def __init__(self, d_k, d_ctx, num_rel, num_ent,
                 gamma: float = 1.0, eta: float = 1.0,
                 temperature: float = 1.0):
        super().__init__()
        self.ctx_scorer = ContextScorer(d_q=d_k, d_ctx=d_ctx,
                                        num_rel=num_rel, num_ent=num_ent)
        self.time_scorer = TimeScorer()
        self.gamma = gamma
        self.eta = eta
        self.temperature = temperature

    def forward(self, Q,            # [B, T, d_k]
                K_kb, V_kb,        # [B, T, K, d_k/d_v]
                ctx_vec, rel_id, ent_id,  # [B, T, K, ...]
                q_min, q_max,      # [B, T] 问题时间（没有时可以给极大范围）
                tau_min, tau_max   # [B, T, K]
                ):
        # 语义相似
        sem = torch.einsum("btd,btkd->btk", Q, K_kb) / math.sqrt(Q.size(-1))
        # 上下文得分
        ctx_score = self.ctx_scorer(Q, ctx_vec, rel_id, ent_id)
        # 时态得分
        time_score = self.time_scorer(q_min, q_max, tau_min, tau_max)

        s = sem + self.gamma * ctx_score + self.eta * time_score
        s = s - s.max(dim=-1, keepdim=True).values
        alpha = torch.softmax(s / self.temperature, dim=-1)  # [B, T, K]

        B, T, K, d_v = V_kb.shape
        V_tilde = torch.bmm(
            alpha.view(B*T, 1, K),
            V_kb.view(B*T, K, d_v),
        ).view(B, T, d_v)
        return alpha, V_tilde
```

### 6.3 KBFusionLayer：并行注意力 + 门控

`kblampp/fusion.py`：

```python
class KBFusionLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.text_mha = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.kb_mha   = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.beta_head = nn.Linear(d_model, 1)

        # 可选：初始化 bias 为负数，使 beta 初始接近 0
        nn.init.constant_(self.beta_head.bias, -2.0)

    def forward(self, h, V_kb_tilde, attn_mask=None):
        """
        h:          [B, T, d_model] 文本隐藏状态
        V_kb_tilde: [B, T, d_model] 已映射到 d_model 的知识注入向量
        """
        Y_txt, _ = self.text_mha(h, h, h, attn_mask=attn_mask)
        Y_kb, _  = self.kb_mha(h, h, V_kb_tilde, attn_mask=attn_mask)

        beta = torch.sigmoid(self.beta_head(h))  # [B, T, 1]
        Y = Y_txt + beta * Y_kb
        return Y, beta
```

### 6.4 注入 Llama3：InjectionWrapper

`kblampp/injection_wrapper.py`：

- 通过 HuggingFace transformers 加载 Llama 3 8B；
- 在若干层上插入 KBFusionLayer；
- 在 forward 中：
  a. 先跑到注入层；
  b. 对该层输出做 linear_q 得到 (Q)；
  c. ANN 检索 + KBSelector → $V_\text{tilde}$；
  d. linear_v 把 $V_\text{tilde}$ 拉到 d_model；
  e. 调 KBFusionLayer 融合；
  f. 再继续后续层。

## 7. 训练流程与实验脚本

### 7.1 统一数据格式与 DataLoader

`train/dataloader.py`：

- 统一把 QA JSONL 转换为训练样本：
  - input_ids：带 prompt 的问题序列（可以是"问题 + 选项/说明"，按实际任务）；
  - labels：目标答案（可用生成式或 span 抽取式）；
  - question_time：若是时态 QA，提供 [B, T] 的时间窗（无则填默认极大区间）；
  - supporting_facts：如果有（合成 KB、HotpotQA），可用于监督 ($\alpha$)。

### 7.2 阶段 A：冻结 Llama，只训 KB 模块

`train/train_stageA.py` 大致步骤：

```python
# 1. load Llama3 8B Instruct，设 requires_grad=False
# 2. 初始化：
#   - linear_q, linear_v
#   - KBSelector, KBFusionLayer（在指定层插入）
# 3. 加载 KnowledgeIndex 和 KBValueStore
# 4. 训练循环：

for batch in train_loader:
    input_ids = batch["input_ids"].to(device)
    labels    = batch["labels"].to(device)
    q_time    = batch["question_time"].to(device)      # [B, 2] or [B, T, 2]

    outputs = model(
        input_ids=input_ids,
        labels=labels,
        question_time=q_time,
        return_alpha=True,   # 方便 debug
    )
    loss = outputs.loss      # 任务交叉熵

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

- 初期建议：只训练 KB 模块的参数 + linear_q, linear_v，不动 Llama。
- 学习率：5e-4 ~ 1e-4，观察梯度稳定性。

### 7.3 阶段 B：解冻顶层，联合优化

`train/train_stageB.py`：

1. 在阶段 A 的 checkpoint 基础上加载；
2. 解冻 Llama 顶层若干层（如最后 4 层）或加 LoRA adapter；
3. 降低学习率：1e-5 ~ 5e-6；
4. 如有 supporting_facts，可以加辅助损失：
   - 对于路径内的事实，鼓励 ($\alpha_{ij}$) 较大；
   - 对于明显时间不匹配的事实，惩罚 ($\alpha_{ij}$) 过大。

## 8. 评估与可解释性输出

### 8.1 标准指标

`eval/eval_qa.py`：

- **单跳任务**（T-REx/合成单跳）：
  - EM / F1；
- **多跳任务**（HotpotQA/合成多跳）：
  - EM / F1 + 是否覆盖完整 supporting facts；
- **时态任务**（TimeQA/temporal）：
  - EM / F1 + Time@Δ（时间误差不超过 Δ 的比例）。

### 8.2 证据链 JSON 导出

`eval/dump_evidence.py`：

- 对每个问题输出：

```json
{
  "qid": "Q0001",
  "dataset": "synth_world_01",
  "question": "Who is the CEO of BrightSky Robotics in 2019?",
  "answer": "Alice Zhang",
  "pred_answer": "Alice Zhang",
  "top_k_facts": [
    {
      "index": 12345,
      "head": "BrightSky Robotics",
      "relation": "chief executive officer",
      "tail": "Alice Zhang",
      "context": "Alice Zhang became the CEO of BrightSky Robotics on March 2, 2019.",
      "time_window": ["2019-03-02", null],
      "alpha": 0.82
    },
    {
      "index": 67890,
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

- 方便人工检查：模型是否确实利用了正确的事实 & 时间窗。

### 8.3 可视化

`eval/visualize_alpha.py`：

- 对单个样本，读取 top_k_facts；
- 使用 matplotlib 画横向条形图：
  - x 轴：Alpha 权重；
  - y 轴：head + relation + tail（截断显示）；
- 选前 10 条展示。

## 9. 实验配置示例（configs）

### 9.1 合成 world + QA（synth_world.yaml）

```yaml
backbone: "meta-llama/Meta-Llama-3-8B-Instruct"
embedding_model: "text-embedding-ada-002"
d_k: 1024
d_v: 1024
inject_layers: [28, 30, 32]        # 依据 Llama3 depth 调整
K_top: 16
gamma: 1.0
eta: 1.0
temperature: 1.0

train_dataset: "data/qa/synth_world_train.jsonl"
eval_dataset:  "data/qa/synth_world_dev.jsonl"

optimizer:
  type: "adamw"
  lr: 5e-4       # Stage A
  weight_decay: 0.1

batch_size: 8
max_steps: 20000
log_interval: 50
save_interval: 1000
```

### 9.2 T-REx（trex.yaml）

```yaml
backbone: "meta-llama/Meta-Llama-3-8B-Instruct"
embedding_model: "text-embedding-ada-002"
d_k: 1024
d_v: 1024
inject_layers: [28, 30, 32]
K_top: 16
gamma: 0.5
eta: 0.5
temperature: 0.7

train_dataset: "data/qa/trex_train.jsonl"
eval_dataset:  "data/qa/trex_dev.jsonl"
```

### 9.3 HotpotQA（hotpot.yaml）

```yaml
backbone: "meta-llama/Meta-Llama-3-8B-Instruct"
embedding_model: "text-embedding-ada-002"
d_k: 1024
d_v: 1024
inject_layers: [28, 30, 32]
K_top: 32  # 多跳任务需要更多候选
gamma: 0.8
eta: 0.3
temperature: 0.8

# 多跳相关配置
multi_hop:
  enable_gnn: true
  gnn_layers: 2
  gnn_hidden_dim: 512

train_dataset: "data/qa/hotpot_train.jsonl"
eval_dataset:  "data/qa/hotpot_dev.jsonl"

optimizer:
  type: "adamw"
  lr: 3e-4
  weight_decay: 0.1
```

- 可增加 multi-hop 相关的 ablation（不启用/启用局部图 GNN）

### 9.4 TimeQA（timeqa.yaml）

```yaml
backbone: "meta-llama/Meta-Llama-3-8B-Instruct"
embedding_model: "text-embedding-ada-002"
d_k: 1024
d_v: 1024
inject_layers: [28, 30, 32]
K_top: 16
gamma: 0.4
eta: 1.2  # 时态任务加大时间权重
temperature: 0.6

# 时间编码配置
time_encoding:
  d_tau: 64
  freq_bands: 32
  max_period: 10000

train_dataset: "data/qa/timeqa_train.jsonl"
eval_dataset:  "data/qa/timeqa_dev.jsonl"

metrics:
  - "em"
  - "f1"
  - "time@1"  # 1年内正确
  - "time@5"  # 5年内正确
```

- 适当加大 eta（时间项权重），并关注 Time@Δ 指标

## 10. 推理服务与 Demo

### 10.1 在线推理服务

`infer/infer_server.py`：

```python
from flask import Flask, request, jsonify
import torch
from kblampp.injection_wrapper import KBLaMPPWrapper

app = Flask(__name__)
model = None

@app.route('/infer', methods=['POST'])
def infer():
    data = request.json
    question = data['question']
    question_time = data.get('question_time', None)
    
    # 推理
    result = model.generate(
        question=question,
        question_time=question_time,
        max_length=512,
        temperature=0.7
    )
    
    return jsonify({
        'answer': result['answer'],
        'evidence': result['top_evidence'],
        'reasoning_time': result['reasoning_time']
    })

if __name__ == '__main__':
    # 加载模型
    model = KBLaMPPWrapper.load_from_checkpoint(
        "checkpoints/best_model.ckpt",
        config="configs/synth_world.yaml"
    )
    app.run(host='0.0.0.0', port=5000)
```

### 10.2 命令行推理接口

`infer/cli_demo.py`：

```python
def interactive_demo():
    print("KBLaM++ 推理演示")
    print("输入 'quit' 退出")
    
    while True:
        question = input("\n问题: ")
        if question.lower() == 'quit':
            break
            
        # 解析时间信息（如果有）
        time_info = extract_time_from_question(question)
        
        result = model.generate(
            question=question,
            question_time=time_info
        )
        
        print(f"答案: {result['answer']}")
        print("\n支持证据:")
        for i, evidence in enumerate(result['top_evidence'][:3]):
            print(f"{i+1}. {evidence['fact']} (权重: {evidence['alpha']:.3f})")
```

## 11. 性能优化与部署建议

### 11.1 ANN 检索优化

```python
# 使用 FAISS GPU 加速
index = KnowledgeIndex.load("store/index_hnsw", gpu=True)

# 批量查询优化
def batch_query(queries, batch_size=1024):
    results = []
    for i in range(0, len(queries), batch_size):
        batch = queries[i:i+batch_size]
        D, I = index.query(batch, k=K_top)
        results.append((D, I))
    return combine_batch_results(results)
```

### 11.2 内存优化

```python
# 使用内存映射加载大文件
class MemoryMappedStore:
    def __init__(self, root):
        self.K = np.load(f"{root}/K.npy", mmap_mode='r')
        self.V = np.load(f"{root}/V.npy", mmap_mode='r')
        
    def get_batch(self, indices):
        # 只加载需要的批次数据
        return self.K[indices], self.V[indices]
```

### 11.3 模型量化

```python
# 使用 8-bit 量化
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload=True
)

model = KBLaMPPWrapper.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    quantization_config=quantization_config
)
```

## 12. 故障排查与调试

### 12.1 常见问题检查清单

1. **维度不匹配**
   - 检查 $d_k$, $d_v$, $d_{model}$ 是否一致
   - 验证 ANN 索引维度与 K.npy 维度匹配

2. **梯度爆炸/消失**
   - 检查 LayerNorm 位置
   - 验证初始化参数范围
   - 监控梯度范数

3. **ANN 检索质量差**
   - 检查 Key 向量是否归一化
   - 验证相似度计算方式（cosine/L2）
   - 调整 Top-K 大小

### 12.2 调试工具

`debug/debug_utils.py`：

```python
def visualize_attention_flow(model, input_text):
    """可视化注意力流动"""
    with torch.no_grad():
        outputs = model(input_text, return_attention=True)
        
    # 绘制注意力热力图
    plot_attention_heatmap(
        outputs['text_attention'],
        outputs['kb_attention'],
        input_text
    )
    
def check_kb_coverage(question, top_k_facts):
    """检查知识库覆盖情况"""
    relevant_facts = find_relevant_facts(question)
    retrieved_facts = set([f['index'] for f in top_k_facts])
    
    coverage = len(relevant_facts & retrieved_facts) / len(relevant_facts)
    print(f"知识库覆盖度: {coverage:.2%}")
```

## 13. 扩展与未来工作

### 13.1 可扩展架构

```python
class ModularKBLaMPP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.retriever = RetrieverModule(config)
        self.selector = SelectorModule(config) 
        self.fuser = FusionModule(config)
        self.generator = GeneratorModule(config)
        
    def forward(self, input_ids, **kwargs):
        # 模块化前向传播
        retrieved = self.retriever(input_ids)
        selected = self.selector(retrieved)
        fused = self.fuser(input_ids, selected)
        output = self.generator(fused)
        return output
```

### 13.2 多模态扩展

```python
class MultiModalKBLaMPP(KBLaMPPWrapper):
    def __init__(self, config):
        super().__init__(config)
        self.visual_encoder = CLIPVisualEncoder()
        self.multimodal_fuser = CrossModalFusion()
        
    def encode_multimodal_kv(self, image, text):
        visual_feat = self.visual_encoder(image)
        text_feat = self.text_encoder(text)
        return self.multimodal_fuser(visual_feat, text_feat)
```

## 14. 参考文献与相关资源

### 14.1 核心论文

- [1] Lewis, P., et al. "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." NeurIPS 2020.
- [2] Guu, K., et al. "REALM: Retrieval-Augmented Language Model Pre-Training." ICML 2020.
- [3] Xiong, W., et al. "Answering Complex Open-Domain Questions with Multi-Hop Dense Retrieval." ICLR 2021.

### 14.2 相关代码库

- FAISS: https://github.com/facebookresearch/faiss
- HuggingFace Transformers: https://github.com/huggingface/transformers  
- DPR: https://github.com/facebookresearch/DPR

### 14.3 数据集链接

- T-REx: https://hadyelsahar.github.io/t-rex/
- HotpotQA: https://hotpotqa.github.io/
- TimeQA: https://github.com/jgc128/timeqa

---

## 附录

### A. 数学符号表

| 符号 | 含义 | 说明 |
|------|------|------|
| $T_i$ | 五元组 | $(h_i, r_i, t_i, c_i, \tau_i)$ |
| $K_i$ | Key 向量 | 用于检索的表示 |
| $V_i$ | Value 向量 | 用于注入的知识表示 |
| $Q_j$ | 查询向量 | 第 j 个 token 的查询 |
| $\alpha_{ij}$ | 注意力权重 | 第 i 个事实对第 j 个 token 的权重 |
| $\beta_j$ | 门控权重 | 控制知识注入强度 |
| $\gamma$ | 上下文权重 | 平衡语义和上下文得分 |
| $\eta$ | 时间权重 | 平衡时间匹配得分 |

### B. 超参数调优指南

1. **学习率**
   - Stage A: 1e-4 ~ 5e-4
   - Stage B: 1e-5 ~ 5e-6
   - LoRA: 1e-3 ~ 1e-4

2. **温度参数**
   - 初始: 1.0
   - 精细调整: 0.5 ~ 2.0
   - 低温度 → 更尖锐的分布

3. **Top-K 选择**
   - 单跳任务: 8 ~ 16
   - 多跳任务: 16 ~ 32  
   - 大规模 KB: 32 ~ 64

### C. 硬件需求估算

| 组件 | 训练阶段 | 最小显存 | 推荐显存 |
|------|----------|----------|----------|
| Llama3 8B | Stage A | 16GB | 24GB+ |
| Llama3 8B | Stage B | 24GB | 40GB+ |
| ANN 索引 | 推理 | 2-8GB | 8-16GB |
| KV Store | 推理 | 4-16GB | 16-32GB |

**注意**: 以上为估算值，实际需求取决于批次大小、序列长度和知识库规模。