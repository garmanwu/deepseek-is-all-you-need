# DeepSeek 论文与代码精讲

一套中文视频课程：从亲手训过的 nanoGPT（GPT-3 Small）出发，一步步读懂 DeepSeek 从 LLM、V2、V3、R1 到 V3.2、V4、V4.1 的论文和开源代码。讲解风格参考 3Blue1Brown，用动画把公式和数字讲清楚。

- **适合谁**：学过 Transformer 基本结构、训过或读过 nanoGPT，想读懂工业界大模型论文的人
- **每课结构**：核心问题 → 补基础 → 动画讲解 → 论文 → 代码 → 延伸阅读
- **规模**：37 课。主线 22 课（★），按顺序看完能读懂 V3 → R1 → V3.2 → V4 → V4.1；选修 15 课（○），可以插在任意一篇之后
- **发布**：视频发布在 B 站（链接见下方课程目录，陆续更新）

## 课程目录


### 0 导论

- ★ **L00 从你的 GPT-3 Small 到 DeepSeek-V4：一张地图**  
  工业界论文看不懂，差的不是数学，是"规模带来的新问题"。从 124M 到 1.6T 参数，冒出了哪几类新问题？

### 1 基础模型与 MoE

- ★ **L01 DeepSeek LLM：放大版 nanoGPT 与 Scaling Law**  
  算力固定时，参数和数据该怎么分？为什么不能只加参数？  
  论文：[2401.02954](https://arxiv.org/abs/2401.02954)
- ★ **L02 MoE 入门：把一个大 FFN 拆成很多小专家**  
  总参数变多了，为什么每个 token 的计算量不一定跟着变多？  
  论文：[2401.06066](https://arxiv.org/abs/2401.06066)
- ★ **L03 负载均衡：不加辅助损失，专家怎么不偷懒**  
  总有几个专家被挤爆，其他专家闲着。能不能不改主损失就把负载分匀？  
  论文：[2408.15664](https://arxiv.org/abs/2408.15664)
- ○ **L04 ESFT：只微调和任务相关的专家**  
  模型每次只用少数专家，微调能不能也只动这几个？  
  论文：[2407.01906](https://arxiv.org/abs/2407.01906)

### 2 KV cache 与 MLA

- ★ **L05 先算账：KV cache 有多大，MLA 怎么压缩它**（已制作）  
  模型装得下，为什么还服务不了长对话？  
  论文：[2405.04434](https://arxiv.org/abs/2405.04434)
- ★ **L06 V3 架构总装：逐行读 model.py**  
  把 MLA、MoE、MTP 拼在一起，一个 token 从进入模型到输出经过了什么？  
  论文：[2412.19437](https://arxiv.org/abs/2412.19437)

### 3 训练系统

- ★ **L07 并行训练入门：DP/TP/PP/EP 与 DualPipe**  
  算子快了，为什么整次训练还在等通信？  
  论文：[2412.19437](https://arxiv.org/abs/2412.19437)
- ★ **L08 专家之间怎么传数据：All-to-All 与 DeepEP**  
  专家分布在不同 GPU 上，token 要被"寄"过去再"寄"回来，这一步怎么不拖后腿？  
  论文：[2412.19437](https://arxiv.org/abs/2412.19437)
- ★ **L09 FP8 训练：用 8 位数字训练 671B 模型**  
  位数越少越快越省显存，但精度怎么保住？  
  论文：[2412.19437](https://arxiv.org/abs/2412.19437)
- ○ **L10 集群与存储：Fire-Flyer、3FS、smallpond**  
  更贵的单卡，一定带来更划算的集群吗？上万张卡读数据，瓶颈在哪？  
  论文：[2408.14158](https://arxiv.org/abs/2408.14158)
- ○ **L11 回看硬件：V3 团队对芯片的期待**  
  瓶颈到底在计算、显存，还是跨节点带宽？  
  论文：[2505.09343](https://arxiv.org/abs/2505.09343)

### 4 后训练与强化学习

- ★ **L12 补课：从交叉熵到策略梯度**  
  模型已经会算概率了，奖励怎样改变下一次生成的分布？  
  论文：[2402.03300](https://arxiv.org/abs/2402.03300)
- ★ **L13 GRPO：没有价值网络的 PPO**  
  同一道题采样 8 个答案、互相比较，为什么就能代替价值网络？  
  论文：[2402.03300](https://arxiv.org/abs/2402.03300)
- ★ **L14 R1-Zero 与 R1：只给最终奖励，推理能力从哪来**  
  没有一步步的示范，只看最终答案对不对，模型怎么学会长链推理？  
  论文：[2501.12948](https://arxiv.org/abs/2501.12948)
- ○ **L15 奖励模型也能"多想几次"：GRM 与推理时扩展**  
  奖励模型判断一次不可靠时，多判几次怎样才真的有用？  
  论文：[2504.02495](https://arxiv.org/abs/2504.02495)
- ○ **L16 形式化证明：Prover V1 → V1.5 → V2**  
  怎样把"看起来对"变成程序能检查的证明？一个难证明怎么拆成子目标？  
  论文：[2405.14333](https://arxiv.org/abs/2405.14333)；[2408.08152](https://arxiv.org/abs/2408.08152)；[2504.21801](https://arxiv.org/abs/2504.21801)
- ○ **L17 答对了不等于证对了：DeepSeekMath-V2 自我验证**  
  答案正确，推理过程仍可能有漏洞。模型怎么检查自己的证明？  
  论文：[2511.22570](https://arxiv.org/abs/2511.22570)

### 5 长上下文与新架构

- ★ **L18 注意力为什么贵：从 FlashAttention 到 NSA**  
  每个 token 都必须精读全部历史吗？  
  论文：[2502.11089](https://arxiv.org/abs/2502.11089)
- ★ **L19 DSA：V3.2 的"闪电索引器"**  
  先用一个很便宜的小打分器选出 top-k 历史 token，再做完整注意力，这样划算吗？  
  论文：[2512.02556](https://arxiv.org/abs/2512.02556)
- ★ **L20 残差流与优化器：mHC 和 Muon**  
  更灵活的跨层连接为什么会让训练不稳？优化器除了 AdamW 还有什么选择？  
  论文：[2512.24880](https://arxiv.org/abs/2512.24880)
- ★ **L21 Engram：查表也是一种稀疏**  
  熟悉的局部模式（比如固定搭配），每次都值得用神经网络重新算一遍吗？  
  论文：[2601.07372](https://arxiv.org/abs/2601.07372)
- ★ **L22 V4：百万 token 上下文的 CSA + HCA**  
  上下文扩大到 100 万 token 时，哪些成本随长度增长，哪些仍由参数决定？  
  论文：[2606.19348](https://arxiv.org/abs/2606.19348)
- ★ **L23 V4.1-Flash：prefill 和 decode 用不同的计算量**  
  同一个模型，为什么读输入（prefill）只激活 8B 参数，生成输出（decode）激活 16B？KV cache 怎么压到每 token 890 字节？  
  论文：[2609.19969](https://arxiv.org/abs/2609.19969)

### 6 推理部署

- ★ **L24 推理系统入门：prefill、decode 与 PD 分离**  
  为什么生成阶段 GPU 常常"吃不饱"？DeepSeek 怎么用几百张卡一起服务一个模型？  
  论文：[2412.19437](https://arxiv.org/abs/2412.19437)
- ★ **L25 推测解码：从 MTP 到 DSpark**  
  一次多猜几个 token，为什么不一定更快？  
  论文：[2607.05147](https://arxiv.org/abs/2607.05147)
- ○ **L26 算子工程：为什么要自己写 kernel**  
  PyTorch 已经很快了，DeepSeek 为什么还要写一堆 CUDA / TileLang 算子？
- ○ **L27 KV cache 的存储瓶颈：DualPath**  
  GPU 明明有空闲，请求为什么还卡在加载历史缓存上？  
  论文：[2602.21548](https://arxiv.org/abs/2602.21548)
- ○ **L28 通读生产推理框架 DeepInfer**  
  把前面讲过的所有技术装进一个真正上线的推理引擎，代码长什么样？  
  论文：[2609.19969](https://arxiv.org/abs/2609.19969)

### 7 多模态

- ○ **L29 视觉语言理解：DeepSeek-VL 与 VL2**  
  一张高分辨率图怎样变成语言模型能处理的 token？长图、宽图怎么切块？  
  论文：[2403.05525](https://arxiv.org/abs/2403.05525)；[2412.10302](https://arxiv.org/abs/2412.10302)
- ○ **L30 理解与生成一体：Janus 系列**  
  理解一张图和画出一张图，为什么需要不同的视觉表示？  
  论文：[2410.13848](https://arxiv.org/abs/2410.13848)；[2411.07975](https://arxiv.org/abs/2411.07975)；[2501.17811](https://arxiv.org/abs/2501.17811)
- ○ **L31 OCR：用一张图压缩一页字**  
  用更少的视觉 token 表示一页字，会丢掉什么？双栏文档按什么顺序读？  
  论文：[2510.18234](https://arxiv.org/abs/2510.18234)；[2601.20552](https://arxiv.org/abs/2601.20552)
- ○ **L32 附：DreamCraft3D 三维生成**  
  论文：[2310.16818](https://arxiv.org/abs/2310.16818)

### 8 代码与 Agent

- ○ **L33 代码模型：Coder、Coder-V2 与 FIM**  
  代码补全为什么不能只练"接着往下写"？  
  论文：[2401.14196](https://arxiv.org/abs/2401.14196)；[2406.11931](https://arxiv.org/abs/2406.11931)
- ★ **L34 Agent 后训练：任务合成与 DSec 沙盒**  
  只生成正确答案，和在工具环境里把任务做完，训练反馈有什么不同？上万个沙盒怎么管？  
  论文：[2512.02556](https://arxiv.org/abs/2512.02556)；[2609.22978](https://arxiv.org/abs/2609.22978)
- ○ **L35 Agent Harness：插件怎么装得上、卸得干净**  
  插件卸载以后，它留下的副作用怎样撤销？  
  论文：[2608.25512](https://arxiv.org/abs/2608.25512)

### 收官

- ★ **L36 回顾：DeepSeek 的设计哲学**  
  把 36 节课串起来——DeepSeek 一直在省什么？（每 token 的计算、KV 显存、通信、数值位数、存储带宽）  
  论文：[2505.09343](https://arxiv.org/abs/2505.09343)

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
