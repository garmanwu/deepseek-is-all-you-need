# DeepSeek 论文与代码精讲

一套中文视频课程：从亲手训过的 nanoGPT（GPT-3 Small）出发，一步步读懂 DeepSeek 从 LLM、V2、V3、R1 到 V3.2、V4、V4.1 的论文和开源代码。讲解风格参考 3Blue1Brown，用动画把公式和数字讲清楚。

- **适合谁**：学过 Transformer 基本结构、训过或读过 nanoGPT，想读懂工业界大模型论文的人
- **每课结构**：核心问题 → 补基础 → 动画讲解 → 论文 → 代码 → 延伸阅读
- **关键资料**：每课列出 1–3 个最核心的来源。论文精确到章节，代码精确到文件和行号，链接指向固定的提交版本，日后官方仓库更新也不会失效
- **规模**：37 课。主线 22 课（★），按顺序看完能读懂 V3 → R1 → V3.2 → V4 → V4.1；选修 15 课（○），可以插在任意一篇之后
- **发布**：视频发布在 B 站（链接见下方课程目录，陆续更新）

## 课程目录


### 0 导论

- ★ **L00 从你的 GPT-3 Small 到 DeepSeek-V4：一张地图**  
  工业界论文看不懂，差的不是数学，是"规模带来的新问题"。从 124M 到 1.6T 参数，冒出了哪几类新问题？  
  关键资料：[Insights into V3](https://arxiv.org/abs/2505.09343)（官方对 V3 设计的复盘，可当全景图）；[DeepSeek-V3 §1](https://arxiv.org/abs/2412.19437)

### 1 基础模型与 MoE

- ★ **L01 DeepSeek LLM：放大版 nanoGPT 与 Scaling Law**  
  算力固定时，参数和数据该怎么分？为什么不能只加参数？  
  关键资料：[DeepSeek LLM §3](https://arxiv.org/abs/2401.02954)（Scaling Laws）；[DeepSeek LLM §2.2](https://arxiv.org/abs/2401.02954)（LLaMA 式结构）
- ★ **L02 MoE 入门：把一个大 FFN 拆成很多小专家**  
  总参数变多了，为什么每个 token 的计算量不一定跟着变多？  
  关键资料：[DeepSeekMoE §3](https://arxiv.org/abs/2401.06066)（细粒度专家与共享专家）；[DeepSeek-V3 §2.1.2](https://arxiv.org/abs/2412.19437)；[DeepSeek-V3/inference/model.py:535](https://github.com/deepseek-ai/DeepSeek-V3/blob/9b4e9788e4a3a731f7567338ed15d3ec549ce03b/inference/model.py#L535)
- ★ **L03 负载均衡：不加辅助损失，专家怎么不偷懒**  
  总有几个专家被挤爆，其他专家闲着。能不能不改主损失就把负载分匀？  
  关键资料：[DeepSeek-V2 §2.2.2–§2.2.4](https://arxiv.org/abs/2405.04434)（V2 的三种均衡损失）；[Loss-Free 均衡 §3](https://arxiv.org/abs/2408.15664)（不加辅助损失的偏置均衡）；[DeepSeek-V3/inference/model.py:535](https://github.com/deepseek-ai/DeepSeek-V3/blob/9b4e9788e4a3a731f7567338ed15d3ec549ce03b/inference/model.py#L535)（看 bias 如何参与 top-k）
- ○ **L04 ESFT：只微调和任务相关的专家**  
  模型每次只用少数专家，微调能不能也只动这几个？  
  关键资料：[ESFT §3](https://arxiv.org/abs/2407.01906)；[ESFT/esft.py](https://github.com/deepseek-ai/ESFT/blob/3746ca74410c92163cc521571dbd17579bfdc962/esft.py)

### 2 KV cache 与 MLA

- ★ **L05 先算账：KV cache 有多大，MLA 怎么压缩它**（已制作）  
  模型装得下，为什么还服务不了长对话？  
  关键资料：[DeepSeek-V2 §2.1](https://arxiv.org/abs/2405.04434)（低秩压缩、矩阵吸收、解耦 RoPE）；[DeepSeek-V3/inference/model.py:396](https://github.com/deepseek-ai/DeepSeek-V3/blob/9b4e9788e4a3a731f7567338ed15d3ec549ce03b/inference/model.py#L396)（naive 与 absorb 两个分支）；[Insights into V3 §2.1.2](https://arxiv.org/abs/2505.09343)（KV cache 横向对比）
- ★ **L06 V3 架构总装：逐行读 model.py**  
  把 MLA、MoE、MTP 拼在一起，一个 token 从进入模型到输出经过了什么？  
  关键资料：[DeepSeek-V3 §2](https://arxiv.org/abs/2412.19437)；[DeepSeek-V3/inference/model.py:696](https://github.com/deepseek-ai/DeepSeek-V3/blob/9b4e9788e4a3a731f7567338ed15d3ec549ce03b/inference/model.py#L696)；[DeepSeek-V3/README_WEIGHTS.md](https://github.com/deepseek-ai/DeepSeek-V3/blob/9b4e9788e4a3a731f7567338ed15d3ec549ce03b/README_WEIGHTS.md)（MTP 模块在权重里，不在 model.py）

### 3 训练系统

- ★ **L07 并行训练入门：DP/TP/PP/EP 与 DualPipe**  
  算子快了，为什么整次训练还在等通信？  
  关键资料：[DeepSeek-V3 §3.2.1](https://arxiv.org/abs/2412.19437)（DualPipe）；[DualPipe/dualpipe/dualpipe.py](https://github.com/deepseek-ai/DualPipe/blob/030ce4325f4ebeb437da4ebc6d00a70469dd58ae/dualpipe/dualpipe.py)
- ★ **L08 专家之间怎么传数据：All-to-All 与 DeepEP**  
  专家分布在不同 GPU 上，token 要被"寄"过去再"寄"回来，这一步怎么不拖后腿？  
  关键资料：[DeepSeek-V3 §3.2.2](https://arxiv.org/abs/2412.19437)（跨节点 All-to-All）；[DeepEP/deep_ep/buffers/ep.py](https://github.com/deepseek-ai/DeepEP/blob/f6aa38fe06cd4a0489e7730365d4e0114970a2f3/deep_ep/buffers/ep.py)
- ★ **L09 FP8 训练：用 8 位数字训练 671B 模型**  
  位数越少越快越省显存，但精度怎么保住？  
  关键资料：[DeepSeek-V3 §3.3](https://arxiv.org/abs/2412.19437)（FP8 训练）；[DeepSeek-V3/inference/kernel.py](https://github.com/deepseek-ai/DeepSeek-V3/blob/9b4e9788e4a3a731f7567338ed15d3ec549ce03b/inference/kernel.py)
- ○ **L10 集群与存储：Fire-Flyer、3FS、smallpond**  
  更贵的单卡，一定带来更划算的集群吗？上万张卡读数据，瓶颈在哪？  
  关键资料：[Fire-Flyer](https://arxiv.org/abs/2408.14158)；[3FS/docs/design_notes.md](https://github.com/deepseek-ai/3FS/blob/22fca04564c7cc230fd8b9523b8b92864e1dad47/docs/design_notes.md)
- ○ **L11 回看硬件：V3 团队对芯片的期待**  
  瓶颈到底在计算、显存，还是跨节点带宽？  
  关键资料：[Insights into V3](https://arxiv.org/abs/2505.09343)；[DeepSeek-V3 §3.5](https://arxiv.org/abs/2412.19437)（对硬件设计的建议）

### 4 后训练与强化学习

- ★ **L12 补课：从交叉熵到策略梯度**  
  模型已经会算概率了，奖励怎样改变下一次生成的分布？  
  关键资料：[DeepSeekMath §5.2.1](https://arxiv.org/abs/2402.03300)（SFT / DPO / PPO / GRPO 统一框架）；[DeepSeekMath §4.1.1](https://arxiv.org/abs/2402.03300)（从 PPO 讲起）
- ★ **L13 GRPO：没有价值网络的 PPO**  
  同一道题采样 8 个答案、互相比较，为什么就能代替价值网络？  
  关键资料：[DeepSeekMath §4.1](https://arxiv.org/abs/2402.03300)（GRPO）；[DeepSeek-V3.2 §3.1](https://arxiv.org/abs/2512.02556)（规模化 GRPO 的修改）
- ★ **L14 R1-Zero 与 R1：只给最终奖励，推理能力从哪来**  
  没有一步步的示范，只看最终答案对不对，模型怎么学会长链推理？  
  关键资料：[DeepSeek-R1 §2–§3](https://arxiv.org/abs/2501.12948)（R1-Zero 与 R1）
- ○ **L15 奖励模型也能"多想几次"：GRM 与推理时扩展**  
  奖励模型判断一次不可靠时，多判几次怎样才真的有用？  
  关键资料：[GRM §3–§4](https://arxiv.org/abs/2504.02495)
- ○ **L16 形式化证明：Prover V1 → V1.5 → V2**  
  怎样把"看起来对"变成程序能检查的证明？一个难证明怎么拆成子目标？  
  关键资料：[Prover-V1.5 §2.3、§3](https://arxiv.org/abs/2408.08152)；[Prover-V2 §2.1](https://arxiv.org/abs/2504.21801)（子目标分解）
- ○ **L17 答对了不等于证对了：DeepSeekMath-V2 自我验证**  
  答案正确，推理过程仍可能有漏洞。模型怎么检查自己的证明？  
  关键资料：[Math-V2 §2](https://arxiv.org/abs/2511.22570)

### 5 长上下文与新架构

- ★ **L18 注意力为什么贵：从 FlashAttention 到 NSA**  
  每个 token 都必须精读全部历史吗？  
  关键资料：[NSA §3.3](https://arxiv.org/abs/2502.11089)（压缩、选择、滑动窗口）
- ★ **L19 DSA：V3.2 的"闪电索引器"**  
  先用一个很便宜的小打分器选出 top-k 历史 token，再做完整注意力，这样划算吗？  
  关键资料：[DeepSeek-V3.2 §2.1](https://arxiv.org/abs/2512.02556)（DSA）；[DeepSeek-V3.2-Exp/inference/model.py:435](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/87e509a2e5a100d221c97df52c6e8be7835f0057/inference/model.py#L435)
- ★ **L20 残差流与优化器：mHC 和 Muon**  
  更灵活的跨层连接为什么会让训练不稳？优化器除了 AdamW 还有什么选择？  
  关键资料：[mHC §4](https://arxiv.org/abs/2512.24880)（mHC）；[DeepSeek-V4 §2.4](https://arxiv.org/abs/2606.19348)（Muon）；[TileKernels/tile_kernels/torch/mhc.py](https://github.com/deepseek-ai/TileKernels/blob/c1cf914a82e6901c679445f872051ce44ac2cc63/tile_kernels/torch/mhc.py)
- ★ **L21 Engram：查表也是一种稀疏**  
  熟悉的局部模式（比如固定搭配），每次都值得用神经网络重新算一遍吗？  
  关键资料：[Engram §2–§3](https://arxiv.org/abs/2601.07372)；[Engram/engram_demo_v1.py](https://github.com/deepseek-ai/Engram/blob/fb7f84a21f91223715394a33a1dc24bbfb7f788e/engram_demo_v1.py)
- ★ **L22 V4：百万 token 上下文的 CSA + HCA**  
  上下文扩大到 100 万 token 时，哪些成本随长度增长，哪些仍由参数决定？  
  关键资料：[DeepSeek-V4 §2.3](https://arxiv.org/abs/2606.19348)（CSA 与 HCA）；[DeepInfer/deepinfer/model_executor/models/v4_modules/compress.py](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/deepinfer/model_executor/models/v4_modules/compress.py)
- ★ **L23 V4.1-Flash：prefill 和 decode 用不同的计算量**  
  同一个模型，为什么读输入（prefill）只激活 8B 参数，生成输出（decode）激活 16B？KV cache 怎么压到每 token 890 字节？  
  关键资料：[V4.1-Flash §2.2–§2.3](https://arxiv.org/abs/2609.19969)（CED 与 CSA2）；[DeepInfer/deepinfer/model_executor/models/v4_modules/cross_layer.py](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/deepinfer/model_executor/models/v4_modules/cross_layer.py)

### 6 推理部署

- ★ **L24 推理系统入门：prefill、decode 与 PD 分离**  
  为什么生成阶段 GPU 常常"吃不饱"？DeepSeek 怎么用几百张卡一起服务一个模型？  
  关键资料：[DeepSeek-V3 §3.4](https://arxiv.org/abs/2412.19437)（prefill 与 decode）；[open-infra-index/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md](https://github.com/deepseek-ai/open-infra-index/blob/56d86855fcf6e08fdfd45ce6280bd24322c93351/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md)（官方推理系统概览）
- ★ **L25 推测解码：从 MTP 到 DSpark**  
  一次多猜几个 token，为什么不一定更快？  
  关键资料：[DSpark §3](https://arxiv.org/abs/2607.05147)（DSpark）；[DeepSeek-V3 §2.2](https://arxiv.org/abs/2412.19437)（MTP）；[DeepSpec/README.md](https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/README.md)
- ○ **L26 算子工程：为什么要自己写 kernel**  
  PyTorch 已经很快了，DeepSeek 为什么还要写一堆 CUDA / TileLang 算子？  
  关键资料：[DeepSeek-V3/inference/kernel.py](https://github.com/deepseek-ai/DeepSeek-V3/blob/9b4e9788e4a3a731f7567338ed15d3ec549ce03b/inference/kernel.py)（Triton 入门）；[FlashMLA/docs/20250422-new-kernel-deep-dive.md](https://github.com/deepseek-ai/FlashMLA/blob/0f11e178da1f67becb087a324b400854931ec3e3/docs/20250422-new-kernel-deep-dive.md)；[DeepSeek-V4 §3.2](https://arxiv.org/abs/2606.19348)（TileLang）
- ○ **L27 KV cache 的存储瓶颈：DualPath**  
  GPU 明明有空闲，请求为什么还卡在加载历史缓存上？  
  关键资料：[DualPath](https://arxiv.org/abs/2602.21548)
- ○ **L28 通读生产推理框架 DeepInfer**  
  把前面讲过的所有技术装进一个真正上线的推理引擎，代码长什么样？  
  关键资料：[DeepInfer/README.md](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/README.md)；[V4.1-Flash §3.2](https://arxiv.org/abs/2609.19969)（推理系统）

### 7 多模态

- ○ **L29 视觉语言理解：DeepSeek-VL 与 VL2**  
  一张高分辨率图怎样变成语言模型能处理的 token？长图、宽图怎么切块？  
  关键资料：[DeepSeek-VL2 §2](https://arxiv.org/abs/2412.10302)；[DeepSeek-VL2/README.md](https://github.com/deepseek-ai/DeepSeek-VL2/blob/ef9f91e2b6426536b83294c11742c27be66361b1/README.md)
- ○ **L30 理解与生成一体：Janus 系列**  
  理解一张图和画出一张图，为什么需要不同的视觉表示？  
  关键资料：[Janus §3.1](https://arxiv.org/abs/2410.13848)；[Janus-Pro](https://arxiv.org/abs/2501.17811)；[Janus/README.md](https://github.com/deepseek-ai/Janus/blob/1daa72fa409002d40931bd7b36a9280362469ead/README.md)
- ○ **L31 OCR：用一张图压缩一页字**  
  用更少的视觉 token 表示一页字，会丢掉什么？双栏文档按什么顺序读？  
  关键资料：[DeepSeek-OCR §3.2](https://arxiv.org/abs/2510.18234)（DeepEncoder）；[DeepSeek-OCR 2](https://arxiv.org/abs/2601.20552)
- ○ **L32 附：DreamCraft3D 三维生成**  
  关键资料：[DreamCraft3D](https://arxiv.org/abs/2310.16818)；[DreamCraft3D/README.md](https://github.com/deepseek-ai/DreamCraft3D/blob/5829ef116d36c871ce2b9e54a6153dd3856a1561/README.md)

### 8 代码与 Agent

- ○ **L33 代码模型：Coder、Coder-V2 与 FIM**  
  代码补全为什么不能只练"接着往下写"？  
  关键资料：[DeepSeek-Coder §3.1.2](https://arxiv.org/abs/2401.14196)（FIM）；[Coder-V2](https://arxiv.org/abs/2406.11931)
- ★ **L34 Agent 后训练：任务合成与 DSec 沙盒**  
  只生成正确答案，和在工具环境里把任务做完，训练反馈有什么不同？上万个沙盒怎么管？  
  关键资料：[DeepSeek-V3.2 §3.2](https://arxiv.org/abs/2512.02556)（在工具调用中思考）；[DSec](https://arxiv.org/abs/2609.22978)（Agent 沙盒）
- ○ **L35 Agent Harness：插件怎么装得上、卸得干净**  
  插件卸载以后，它留下的副作用怎样撤销？  
  关键资料：[Harness 论文 §1.2、§3](https://arxiv.org/abs/2608.25512)

### 收官

- ★ **L36 回顾：DeepSeek 的设计哲学**  
  把 36 节课串起来——DeepSeek 一直在省什么？（每 token 的计算、KV 显存、通信、数值位数、存储带宽）  
  关键资料：[Insights into V3](https://arxiv.org/abs/2505.09343)

## 一手资料

### 论文（arXiv）

- [2310.16818](https://arxiv.org/abs/2310.16818) DreamCraft3D: Hierarchical 3D Generation with Bootstrapped Diffusion Prior
- [2401.02954](https://arxiv.org/abs/2401.02954) DeepSeek LLM: Scaling Open-Source Language Models with Longtermism
- [2401.06066](https://arxiv.org/abs/2401.06066) DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models
- [2401.14196](https://arxiv.org/abs/2401.14196) DeepSeek-Coder: When the Large Language Model Meets Programming -- The Rise of Code Intelligence
- [2402.03300](https://arxiv.org/abs/2402.03300) DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models
- [2403.05525](https://arxiv.org/abs/2403.05525) DeepSeek-VL: Towards Real-World Vision-Language Understanding
- [2405.04434](https://arxiv.org/abs/2405.04434) DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model
- [2405.14333](https://arxiv.org/abs/2405.14333) DeepSeek-Prover: Advancing Theorem Proving in LLMs through Large-Scale Synthetic Data
- [2406.11931](https://arxiv.org/abs/2406.11931) DeepSeek-Coder-V2: Breaking the Barrier of Closed-Source Models in Code Intelligence
- [2407.01906](https://arxiv.org/abs/2407.01906) Let the Expert Stick to His Last: Expert-Specialized Fine-Tuning for Sparse Architectural Large Language Models
- [2408.08152](https://arxiv.org/abs/2408.08152) DeepSeek-Prover-V1.5: Harnessing Proof Assistant Feedback for Reinforcement Learning and Monte-Carlo Tree Search
- [2408.14158](https://arxiv.org/abs/2408.14158) Fire-Flyer AI-HPC: A Cost-Effective Software-Hardware Co-Design for Deep Learning
- [2408.15664](https://arxiv.org/abs/2408.15664) Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts
- [2410.13848](https://arxiv.org/abs/2410.13848) Janus: Decoupling Visual Encoding for Unified Multimodal Understanding and Generation
- [2411.07975](https://arxiv.org/abs/2411.07975) JanusFlow: Harmonizing Autoregression and Rectified Flow for Unified Multimodal Understanding and Generation
- [2412.10302](https://arxiv.org/abs/2412.10302) DeepSeek-VL2: Mixture-of-Experts Vision-Language Models for Advanced Multimodal Understanding
- [2412.19437](https://arxiv.org/abs/2412.19437) DeepSeek-V3 Technical Report
- [2501.12948](https://arxiv.org/abs/2501.12948) DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning
- [2501.17811](https://arxiv.org/abs/2501.17811) Janus-Pro: Unified Multimodal Understanding and Generation with Data and Model Scaling
- [2502.11089](https://arxiv.org/abs/2502.11089) Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention
- [2504.02495](https://arxiv.org/abs/2504.02495) Inference-Time Scaling for Generalist Reward Modeling
- [2504.21801](https://arxiv.org/abs/2504.21801) DeepSeek-Prover-V2: Advancing Formal Mathematical Reasoning via Reinforcement Learning for Subgoal Decomposition
- [2505.09343](https://arxiv.org/abs/2505.09343) Insights into DeepSeek-V3: Scaling Challenges and Reflections on Hardware for AI Architectures
- [2510.18234](https://arxiv.org/abs/2510.18234) DeepSeek-OCR: Contexts Optical Compression
- [2511.22570](https://arxiv.org/abs/2511.22570) DeepSeekMath-V2: Towards Self-Verifiable Mathematical Reasoning
- [2512.02556](https://arxiv.org/abs/2512.02556) DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models
- [2512.24880](https://arxiv.org/abs/2512.24880) mHC: Manifold-Constrained Hyper-Connections
- [2601.07372](https://arxiv.org/abs/2601.07372) Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models
- [2601.20552](https://arxiv.org/abs/2601.20552) DeepSeek-OCR 2: Visual Causal Flow
- [2602.21548](https://arxiv.org/abs/2602.21548) DualPath: Breaking the Storage Bandwidth Bottleneck in Agentic LLM Inference
- [2606.19348](https://arxiv.org/abs/2606.19348) DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence
- [2607.05147](https://arxiv.org/abs/2607.05147) DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation
- [2608.25512](https://arxiv.org/abs/2608.25512) A Programming Paradigm for Spatiotemporal Composability
- [2609.19969](https://arxiv.org/abs/2609.19969) DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression
- [2609.22978](https://arxiv.org/abs/2609.22978) DeepSeek Elastic Compute (DSec): A Sandbox Infrastructure for Effective Agentic Training at Scale

### 官方代码仓库（github.com/deepseek-ai）

课程中逐行讲解的代码均来自 DeepSeek 官方开源仓库：

[3FS](https://github.com/deepseek-ai/3FS)、[awesome-deepseek-agent](https://github.com/deepseek-ai/awesome-deepseek-agent)、[awesome-deepseek-coder](https://github.com/deepseek-ai/awesome-deepseek-coder)、[awesome-deepseek-integration](https://github.com/deepseek-ai/awesome-deepseek-integration)、[clangd-ascend](https://github.com/deepseek-ai/clangd-ascend)、[DeepEP](https://github.com/deepseek-ai/DeepEP)、[DeepEP-Ascend](https://github.com/deepseek-ai/DeepEP-Ascend)、[DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)、[DeepGEMM-Ascend](https://github.com/deepseek-ai/DeepGEMM-Ascend)、[DeepInfer](https://github.com/deepseek-ai/DeepInfer)、[DeepJIT](https://github.com/deepseek-ai/DeepJIT)、[DeepSeek-Coder](https://github.com/deepseek-ai/DeepSeek-Coder)、[DeepSeek-Coder-V2](https://github.com/deepseek-ai/DeepSeek-Coder-V2)、[deepseek-harness](https://github.com/deepseek-ai/deepseek-harness)、[DeepSeek-LLM](https://github.com/deepseek-ai/DeepSeek-LLM)、[DeepSeek-Math](https://github.com/deepseek-ai/DeepSeek-Math)、[DeepSeek-Math-V2](https://github.com/deepseek-ai/DeepSeek-Math-V2)、[DeepSeek-MoE](https://github.com/deepseek-ai/DeepSeek-MoE)、[DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR)、[DeepSeek-OCR-2](https://github.com/deepseek-ai/DeepSeek-OCR-2)、[DeepSeek-Prover-V1.5](https://github.com/deepseek-ai/DeepSeek-Prover-V1.5)、[DeepSeek-Prover-V2](https://github.com/deepseek-ai/DeepSeek-Prover-V2)、[DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1)、[deepseek-recipe](https://github.com/deepseek-ai/deepseek-recipe)、[DeepSeek-V2](https://github.com/deepseek-ai/DeepSeek-V2)、[DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3)、[DeepSeek-V3.2-Exp](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp)、[DeepSeek-VL](https://github.com/deepseek-ai/DeepSeek-VL)、[DeepSeek-VL2](https://github.com/deepseek-ai/DeepSeek-VL2)、[DeepSelect](https://github.com/deepseek-ai/DeepSelect)、[DeepSpec](https://github.com/deepseek-ai/DeepSpec)、[DreamCraft3D](https://github.com/deepseek-ai/DreamCraft3D)、[dsh-libreoffice-kit](https://github.com/deepseek-ai/dsh-libreoffice-kit)、[dsh-node-addon-require-builtin](https://github.com/deepseek-ai/dsh-node-addon-require-builtin)、[DualPipe](https://github.com/deepseek-ai/DualPipe)、[Engram](https://github.com/deepseek-ai/Engram)、[EPLB](https://github.com/deepseek-ai/EPLB)、[ESFT](https://github.com/deepseek-ai/ESFT)、[FlashMLA](https://github.com/deepseek-ai/FlashMLA)、[Janus](https://github.com/deepseek-ai/Janus)、[LPLB](https://github.com/deepseek-ai/LPLB)、[open-infra-index](https://github.com/deepseek-ai/open-infra-index)、[profile-data](https://github.com/deepseek-ai/profile-data)、[smallpond](https://github.com/deepseek-ai/smallpond)、[TileKernels](https://github.com/deepseek-ai/TileKernels)

### 官方文章

- [DeepSeek-V3.1 Release | DeepSeek API Docs](https://api-docs.deepseek.com/news/news250821/)
- [DeepSeek-V4 - a deepseek-ai Collection](https://huggingface.co/collections/deepseek-ai/deepseek-v4)
- [DeepSeek | Into the Unknown](https://www.deepseek.com/en/news/)
- [DeepSeek V4.1 Flash 大规模推理部署实践](https://zhuanlan.zhihu.com/p/2095560295930508953)
- [DeepSeek 弹性计算 (DSec)：面向大规模 Agent 训练的沙盒基础设施](https://zhuanlan.zhihu.com/p/2088265189233779592)
- [DeepSeek-V3 / R1 推理系统概览](https://zhuanlan.zhihu.com/p/27181462601)
- [DeepSeek-V4.1-Flash 技术报告 PDF](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/DeepSeek_V41_Tech_Report.pdf)
- [DeepSeek-V4.1-Flash-1017: Wish-Driven Programming Made Real | DeepSeek API Docs](https://api-docs.deepseek.com/news/news261017/)
- [DeepSeek-V3.2 技术报告 PDF](https://huggingface.co/deepseek-ai/DeepSeek-V3.2/resolve/main/assets/paper.pdf)

## 延伸阅读

第三方解读用于帮助理解，其中的推测和评价属于作者观点，以论文和代码为准。

### 中文

- [陈巍：DeepSeek 开源Day（3）DeepGEMM深入分析](https://zhuanlan.zhihu.com/p/26437292382)（陈巍 博士）
- [陈巍：DeepSeek 开源Day（1）-FlashMLA 深入分析](https://zhuanlan.zhihu.com/p/26031898869)（陈巍 博士）
- [陈巍：DeepSeek V3/R1的架构与训练技术2万字长文分析（上）](https://zhuanlan.zhihu.com/p/21208287743)（陈巍 博士）
- [陈巍：DeepSeek 开源Day（6）推理系统概览解读](https://zhuanlan.zhihu.com/p/27264662709)（陈巍 博士）
- [DeepSeek是否有国运级创新？2万字解读与硬核分析V3/R1的架构](https://tech.ifeng.com/c/8gpIvrINs11)
- [陈巍：DeepSeek V3/R1的架构与训练技术2万字长文分析（下）](https://zhuanlan.zhihu.com/p/21755758234)（陈巍 博士）
- [陈巍：DeepSeek 开源Day（5）3FS&smallpond深入分析](https://zhuanlan.zhihu.com/p/26958884790)（陈巍 博士）
- [陈巍：DeepSeek 开源Day（4）DualPipe&EPLB深入分析](https://zhuanlan.zhihu.com/p/26744800336)（陈巍 博士）
- [DeepSeek 开源推理框架：走向真正普惠的未来 – 人人都是产品经理](https://www.woshipm.com/share/6481490.html)
- [陈巍：DeepSeek V3/R1架构的深度分析与深度思考（52页公开报告）](https://www.cdut.edu.cn/__local/4/24/A3/26179C3E81EECC0BBE7660B56D4_5618A1E3_7AC4BF.pdf)
- [陈巍：DeepSeek 开源Day（2）DeepEP深入分析](https://zhuanlan.zhihu.com/p/26204046487)（陈巍 博士）
- [缓存与效果的极限拉扯：从MHA、MQA、GQA到MLA](https://spaces.ac.cn/archives/10091)（苏剑林）
- [Transformer升级之路：2、博采众长的旋转式位置编码](https://spaces.ac.cn/archives/8265)（苏剑林）
- [科学空间 kexue.fm](https://kexue.fm/)（苏剑林）：MoE环游记、Transformer升级之路、低秩近似之路、Muon 系列等，课程大纲在对应各课列出了具体文章

### 英文

- [The Big LLM Architecture Comparison](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison)（Sebastian Raschka）
- [Rotary Embeddings: A Relative Revolution](https://blog.eleuther.ai/rotary-embeddings/)（EleutherAI）
- [Mixture of Experts Explained](https://huggingface.co/blog/moe)（Hugging Face）
- [A Visual Guide to Mixture of Experts (MoE)](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mixture-of-experts)（Maarten Grootendorst）
- [Illustrating Reinforcement Learning from Human Feedback (RLHF)](https://huggingface.co/blog/rlhf)（Hugging Face）
- [Proximal Policy Optimization (Spinning Up)](https://spinningup.openai.com/en/latest/algorithms/ppo.html)（OpenAI）
- [The Illustrated DeepSeek-R1](https://newsletter.languagemodels.co/p/the-illustrated-deepseek-r1)（Jay Alammar）
- [A Visual Guide to Reasoning LLMs](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-reasoning-llms)（Maarten Grootendorst）
- [Getting Started with Distributed Data Parallel](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)（PyTorch）
- [The Ultra-Scale Playbook: Training LLMs on GPU Clusters](https://huggingface.co/spaces/nanotron/ultrascale-playbook)（Hugging Face）

## 制作方式

- 动画：[Manim Community](https://www.manim.community/)，每一章是一个 Scene，旁白和动画按句对齐
- 配音：TTS 合成（试制阶段用 edge-tts，正式版计划换成更自然的音色），字幕和章节时间由程序自动生成
- 版式：一屏一个焦点；固定的语义配色（query 黄、key 蓝、value 绿、问题红）；公式中的每个数字都注明出处（论文表号或代码行号）

## 版权说明

本仓库只包含课程说明和目录。论文、代码和第三方文章的版权归原作者所有，这里只提供原始链接。如需引用课程内容，请注明出处。
