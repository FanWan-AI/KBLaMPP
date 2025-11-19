# KBLaM++ Plan B 实施指南（4090 版本）

本文档基于 `kblampp-planB.md` 与整体理论总览，对 **在单张 RTX 4090 上实施 KBLaM++ Plan B** 给出一份可执行的工程路线图，主要解决：

- 模型如何选择（backbone / embedding）
- 数据路线如何规划（先合成还是先真实数据）
- 在 4090 上的训练 / 推理配置与阶段划分
- 每一步对应的脚本、配置文件和预期产出

目标是：

> 在 3B 级小模型上，完整跑通 KBLaM++ 流程：五元组构建 → K/V 编码 → ANN 检索 → 注入 → 训练 & 评测 → 证据链导出，并为后续扩展到更大模型打好基础。

---

## 一、总体实施阶段

推荐分三大阶段推进，降低风险、快速拿到端到端 Demo：

### 阶段 0：环境与骨架确认（约 1–2 天）

目标：项目主干可运行，依赖清晰，Plan B 默认配置固定下来。

- 明确依赖栈（建议写进 `requirements.txt`）：
  - `transformers`、`accelerate`（或 `deepspeed`，可选）、`peft`、`bitsandbytes`
  - `faiss-cpu`（或其他 ANN 库，如 `hnswlib`）
  - `sentence-transformers` 或直接使用 HF embedding 模型
  - 常规工具库（`numpy` / `pandas` / `pyyaml` 等）
- 确认代码骨架：
  - `offline/`：数据 → 五元组 → K/V → 索引
  - `kblampp/`：`knowledge_index.py`、`kb_store.py`、`selector.py`、`fusion.py`、`injection_wrapper.py`
  - `train/`：`train_stageA.py` 为主训练入口
  - `eval/`、`infer/`：评测与推理服务
- 在 `README.md` 中锁定 Plan B 默认：
  - Backbone：`Qwen/Qwen2.5-3B-Instruct`
  - Embedding：`BAAI/bge-small-en-v1.5`
  - 首个数据源：**合成 world + QA**

### 阶段 1：合成 world + QA 上跑通全流程（优先）

目标：使用 **合成小世界**，在 Qwen2.5‑3B 上完整验证 KBLaM++ 架构与代码路径。

- 使用 LLM API 构造一个小型合成世界（几百～一两千条五元组）以及带 `supporting_facts` 和 `question_time` 的 QA。
- 全部离线跑通：
  - `build_5tuple_from_*.py` / `gen_synth_world.py` / `gen_synth_qa.py`
  - `encode_kv.py`
  - `build_index.py`
- 在线部分：
  - 在 Qwen2.5‑3B 上插入 KBLaM++ 模块（Top‑k 选择 + 并行注意力 + β 门控）。
  - 在合成 QA 上做 Stage A 训练（只训 KB 模块），对比“无 KB”的基线性能。
  - 导出 Top‑k 五元组及其权重（α）作为最小可解释输出。

### 阶段 2：迁移到真实数据子集（Hotpot + TimeQA）

目标：在真实数据上证明 KBLaM++ 的优势，同时保持 Plan B 的显存预算。

- 为 HotpotQA、TimeQA 实现/完善五元组构建脚本：
  - `offline/build_5tuple_from_hotpot.py`（带 `--max_samples`）
  - `offline/build_5tuple_from_timeqa.py`
- 重用 encode / 索引 / 训练流程：
  - 在 `configs/hotpot.yaml`、`configs/timeqa.yaml` 中指定路径和超参。
- 逐步增加：
  - 时态损失（时间窗一致性）
  - 路径损失（利用支持事实 `supporting_facts`）
- 目标：
  - Hotpot / TimeQA 上的 EM/F1 对比原始 LLM 有明显提升。
  - 能导出问题–答案–证据链 JSON。

### 阶段 3：工程化与 Demo（可选接入行业 KB）

目标：形成可以对外展示 / 给业务看的 Demo，并为行业 KB 接入留好接口。

- 打磨 `infer/infer_server.py`：提供 HTTP / CLI 接口。
- 完善可解释性：
  - `eval/dump_evidence.py`：批量导出证据链 JSON。
  - `eval/visualize_alpha.py`：可视化 α 或路径权重。
- 若有行业知识库：
  - 设计五元组构建流水线，对接现有 KB / 文本库。
  - 复用 Plan B 所有离线 / 在线结构。

---

## 二、模型与资源选择（面向 RTX 4090）

### 1. Backbone LLM 选择

**首选主力：`Qwen/Qwen2.5-3B-Instruct`**

- 适合原因：
  - 3B 规模在 24GB 显存 + 4bit 量化 + LoRA 下，训练 / 推理均可接受。
  - 中英通吃，后续扩展中文任务方便。
  - HF 接口标准，`AutoModelForCausalLM` 即可加载。
- 推荐配置（Plan B）：
  - 精度：4-bit 量化（`bnb_4bit`，NF4 + double quant）+ LoRA。
  - 序列长度：`max_seq_len = 512` 起步。
  - 注入层：例如 `inject_layers: [12, 20]`（中层 + 高层）。

**备用轻量：`meta-llama/Llama-3.2-1B-Instruct`**

- 用途：
  - 结构 debug、教学、小显存场景。
  - 当 3B 显存吃紧时，先在 1B 上验证结构与代码正确性。
- 策略：
  - 在 `configs/backbone_llama1b.yaml` 中保持与 3B 相同字段（`d_model`、`inject_layers` 等），只改模型名和维度。

### 2. 句向量 / Embedding 模型

**优先方案：`BAAI/bge-small-en-v1.5`**

- 特点：
  - 维度 `d_ctx = 384`。
  - 质量好、体积小，CPU / GPU 都好用。

**备用方案：`sentence-transformers/all-MiniLM-L6-v2`**

- 维度同样较小（384），英文效果也不错，可作为备选。

在配置文件中的统一字段：

```yaml
embedding_model: "BAAI/bge-small-en-v1.5"
d_ctx: 384
d_k: 384
d_v: 384
```

### 3. 注入结构与超参（Plan B 标准配置）

针对 Qwen2.5‑3B + RTX 4090，推荐一套统一的默认配置：

- 注入层：`inject_layers: [12, 20]`
- Top‑k：`K_top: 16`
- 维度：
  - `d_ctx = 384`
  - `d_k = d_v = 384`
- 门控 / 温度：
  - `gamma = 1.0`
  - `eta = 1.0`
  - `temperature = 1.0`
- 训练 batch 与序列长度：
  - `batch_size: 2`
  - `grad_accum: 4`（等效 batch 8）
  - `max_seq_len: 512`
- 量化与微调：
  - `load_in_4bit: true`
  - Stage A：冻结 backbone，仅训练 KB 模块（Selector + Fusion 等）。
  - Stage B（可选）：在最后 2–4 层加载 LoRA，配合小学习率微调。

---

## 三、数据路线：先合成，后真实数据

### 1. 先用 LLM API 构造合成世界的原因

- 理论层面：
  - 合成 world 可以精确控制：
    - 一跳 / 两跳 / 三跳链路结构；
    - 时间窗（开始 / 结束时间）；
    - 事实间依赖与干扰样本。
  - 支持明确的 `supporting_facts`、`question_time`，非常适合监督 α（门控）与时间模块。
- 工程层面：
  - KB 与 QA 规模可控，显存与训练时间压力小。
  - 在 1–2 周内即可跑出一个“端到端 + 可解释” Demo，用来验证：
    - 公式实现是否正确；
    - 4090 上的显存 / 速度是否可接受；
    - 训练脚本与 config 流程是否顺畅。

### 2. 合成 world + QA 实施方案

#### 2.1 合成 world：`offline/gen_synth_world.py`

- 使用 GPT‑4 / Qwen‑Plus 等 LLM API，根据 `kblampp-planB.md` 中的 prompt 模板生成：
  - `entities`：10–30 个实体。
  - `facts`：200–1000 条五元组。
- 输出格式：
  - 统一存为 JSON / JSONL：`data/5tuple/synth_world_01.jsonl`。
  - 每条记录为：`(h, r, t, c, time_window)`，schema 与理论总览一致。

#### 2.2 合成 QA：`offline/gen_synth_qa.py`

- 输入：上述 `synth_world_01.json`。
- 输出：包含单跳 / 多跳 / 时态问题的 QA 集：
  - `data/qa/synth_world_train.jsonl`
  - `data/qa/synth_world_dev.jsonl`
- 每条样本包含：
  - `qid`、`dataset`
  - `question`、`answer`
  - `type`（`single-hop` / `multi-hop` / `temporal`）
  - `supporting_facts`：使用到的五元组索引（0-based）
  - `question_time`：问题关联的时间窗

#### 2.3 配置文件：`configs/synth_world.yaml`

- 指定数据路径：

```yaml
train_5tuple_path: "data/5tuple/synth_world_01.jsonl"
train_qa_path:     "data/qa/synth_world_train.jsonl"
dev_qa_path:       "data/qa/synth_world_dev.jsonl"
```

- 指定模型与注入参数：

```yaml
backbone: "Qwen/Qwen2.5-3B-Instruct"
embedding_model: "BAAI/bge-small-en-v1.5"
d_ctx: 384
d_k: 384
d_v: 384
inject_layers: [12, 20]
K_top: 16
```

### 3. 再接 HotpotQA / TimeQA 子集

在合成世界版流程跑通后，再接入真实数据：

- 实现 / 完善：
  - `offline/build_5tuple_from_hotpot.py`：
    - 从 Hotpot 原始 json 中抽取：实体、关系、tail、context 句子、时间窗（若有）。
    - 提供 `--max_samples`，例如初始只取 5k 问题构建 KB + QA 子集。
  - `offline/build_5tuple_from_timeqa.py`：
    - 使用 TimeQA 的时间标注构造 `time_window`。
- 配置：
  - 在 `configs/hotpot.yaml`、`configs/timeqa.yaml` 中指定对应路径和注入超参（可复制 `synth_world.yaml` 并修改数据源字段）。
- 训练与评测：
  - 重用 Stage A / Stage B 训练脚本。
  - 逐步增加时态相关损失、路径对比损失，验证真实数据上的收益。

---

## 四、RTX 4090 上的训练与运行配置

### 1. 环境准备

在项目根目录（`kblam_pp/kblam_pp`）下：

```powershell
cd d:\Code\kblam_pp\kblam_pp

python -m venv .venv
.\.venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

> 说明：`requirements.txt` 可由依赖反推生成，建议包含 transformers、faiss-cpu、peft、bitsandbytes、sentence-transformers 等。

### 2. 合成 world 的离线流水线

在完成 `synth_world` 相关脚本与 config 后，推荐的命令顺序：

```powershell
# 1. 生成合成世界（依赖 LLM API）
python .\offline\gen_synth_world.py --config .\configs\synth_world.yaml

# 2. 生成对应 QA
ython .\offline\gen_synth_qa.py --config .\configs\synth_world.yaml

# 3. 对五元组进行 K/V 编码
python .\offline\encode_kv.py --config .\configs\synth_world.yaml

# 4. 构建 ANN 索引（faiss-cpu 等）
python .\offline\build_index.py --config .\configs\synth_world.yaml
```

预期产物：

- `store/K.npy`、`store/V.npy` 等 K/V 向量文件
- `store/index_hnsw/` 或类似目录下的 ANN 索引
- 对应的 meta 信息（时间窗、实体 / 关系 id 等）

### 3. Stage A 训练（Qwen2.5‑3B + 4bit + LoRA）

在 `configs/synth_world.yaml` 中的关键训练字段示例：

```yaml
backbone: "Qwen/Qwen2.5-3B-Instruct"
embedding_model: "BAAI/bge-small-en-v1.5"

inject_layers: [12, 20]
K_top: 16

d_ctx: 384
d_k: 384
d_v: 384

train_dataset: "data/qa/synth_world_train.jsonl"
eval_dataset:  "data/qa/synth_world_dev.jsonl"

batch_size: 2
grad_accum: 4
max_seq_len: 512

optimizer:
  type: "adamw"
  lr: 5e-4
  weight_decay: 0.01
```

训练命令：

```powershell
python .\train\train_stageA.py --config .\configs\synth_world.yaml --device cuda
```

- Stage A：
  - 冻结 backbone 所有参数。
  - 只训练 KB 相关模块：Q 投影、Selector、Fusion、时间得分等。
  - 目标是让 α、β 学会在 QA 任务上“合理使用知识”。

### 4. 可选 Stage B：解冻顶层 / LoRA

在显存允许、Stage A 效果已稳定的前提下，可进行 Stage B：

- 解冻最后 2–4 层，或只通过 LoRA 对这些层进行轻量微调。
- 降低学习率：`lr ~ 1e-5` 量级。
- 适当增加步数 / epoch 数，避免过拟合。
- 重点观察整体 EM/F1 是否进一步提升，以及证据链是否仍然稳定。

### 5. 评测与证据链导出

在训练完成、得到模型 checkpoint 后：

```powershell
# 1. QA 评测
python .\eval\eval_qa.py --config .\configs\synth_world.yaml --checkpoint path\to\model.pt

# 2. 导出证据链 JSON
python .\eval\dump_evidence.py --config .\configs\synth_world.yaml --checkpoint path\to\model.pt

# 3. 可视化 alpha / 路径（若已实现）
python .\eval\visualize_alpha.py --input path\to\evidence.json --output vis\
```

预期：

- EM/F1 明显高于“只用 backbone 不接 KBLaM++”。
- 正确答案对应的事实，在 Top‑k 五元组中具有较高 α 权重。
- 时间窗匹配的事实，有更高的得分和权重。

### 6. 推理 Demo 与 API 服务

建议在 `infer/infer_server.py` 中实现一个最小可用服务：

- 启动命令：

```powershell
python .\infer\infer_server.py --config .\configs\synth_world.yaml --checkpoint path\to\model.pt
```

- 至少提供两个接口：
  - `/qa`：输入 `question`，输出 `answer` + Top‑k 五元组列表（含 α、时间窗）。
  - `/qa_with_paths`：在启用多跳图推理后，输出路径级证据链。

---

## 五、项目 checklist（锁定方案用）

便于团队快速对齐的一份“Plan B 方案确认清单”：

### 1. 模型与资源

- [x] Backbone：`Qwen/Qwen2.5-3B-Instruct`（主力），`Llama-3.2-1B-Instruct`（备用）
- [x] Embedding：`BAAI/bge-small-en-v1.5`，`d_ctx = 384`
- [x] 注入结构：`inject_layers = [12, 20]`，`K_top = 16`，`d_k = d_v = 384`
- [x] 训练配置：batch 2、grad_accum 4、seq_len 512、Stage A 冻结 backbone

### 2. 数据路线

- [x] 第一阶段只用 **合成 world + QA**（`gen_synth_world.py` + `gen_synth_qa.py`）
- [ ] 第二阶段：Hotpot 子集（`build_5tuple_from_hotpot.py --max_samples`）
- [ ] 第三阶段：TimeQA 子集（强调时态一致性）

### 3. 训练路线

- [x] Stage A：冻结 backbone，只训练 KB 模块（Selector / Fusion / TimeScorer 等）
- [ ] Stage B（可选）：解冻顶层 / LoRA 小步微调

### 4. 评测与 Demo

- [x] QA 评测：EM / F1 指标
- [x] 证据链 JSON 导出：问题–答案–Top‑k 五元组 + α + 时间窗
- [ ] 路径可视化：多跳路径图 / HTML 展示
- [ ] 推理服务：`infer_server.py` 暴露 `/qa` 与 `/qa_with_paths` 接口

---

## 六、后续扩展方向（简要）

在 Plan B 成功跑通、效果可接受之后，可以考虑：

- 扩展到更大规模模型（8B / 14B）：只需要在 config 中更换 backbone，并调整 batch / 注入层。
- 加强多跳图推理（ego‑graph + Relational-GAT）：
  - 在 `fusion.py` / `selector.py` 中实现 4.3 节描述的局部图传播与残差融合。
  - 在 eval 与训练中引入路径对比损失与路径指标（Path‑F1、Causal-Δ）。
- 接入真实行业知识库：
  - 将行业 KB 映射为统一五元组表示；
  - 重用 encode / 索引 / 注入与训练流程；
  - 构建针对业务场景的小回归集和可解释性评估。

本实施指南旨在作为团队推进 KBLaM++ Plan B 的“路线图”和对照表，后续实施过程中可以在本文件上持续补充实际参数、实验结果和经验记录。