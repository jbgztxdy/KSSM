# `KSSM`（Kernel State-Space Storage Engine）完整研究档案与全阶段实验报告

- **文档版本**：v2.0（完整全景版，覆盖 `Exp 0-A` → `Exp 4` 全栈闭环实测与统一生产引擎）
- **最后更新**：2026 年 11 月 24 日
- **目标投稿层级**：系统顶会正刊（FAST / OSDI / SOSP）
- **实验平台**：`storage-192`（双路 Intel Xeon Gold 6530 / 128 核 / 稳态基频 `2.10 GHz` / L1d `48 KiB`/核 / L2 `2 MiB`/核 / `~503 GiB` DRAM / Linux `5.15.0-177-generic` / Samsung PM174X 企业级 NVMe `/dev/nvme7n1` 挂载于 `/mnt/nvme12` / GPU 3 `NVIDIA A100-PCIE-40GB`）
- **实验总目录**：`/home/yjw/workspace/research/KSSM/`

---

## 目录导航

- [零、执行摘要与全局进度总览（Go / No-Go 全绿通关）](#零执行摘要与全局进度总览go--no-go-全绿通关)
- [一、研究背景、文献版图与核心灵感](#一研究背景文献版图与核心灵感)
- [二、最终系统愿景、目标与研究意义](#二最终系统愿景目标与研究意义)
- [三、KSSM 模型与系统架构深度设计（数学推导 + 字节级内核实现）](#三kssm-模型与系统架构深度设计数学推导--字节级内核实现)
- [四、离线训练、量化感知微调（Hard-STE QAT）与零 BPTT 慢回路方法论](#四离线训练量化感知微调hard-ste-qat与零-bptt-慢回路方法论)
- [五、实验平台勘察、数据采集体系与严格实验卫生](#五实验平台勘察数据采集体系与严格实验卫生)
- [六、全部对比方法与内部消融设计（确保苹果对苹果公平性）](#六全部对比方法与内部消融设计确保苹果对苹果公平性)
- [七、全阶段实测结果与深入分析（Exp 0-A → Exp 4）](#七全阶段实测结果与深入分析exp-0-a--exp-4)
- [八、真实科研排坑实录（12 个关键系统/算法陷阱与根治方案）](#八真实科研排坑实录12-个关键系统算法陷阱与根治方案)
- [九、必须正视的现存局限与审稿风险预判](#九必须正视的现存局限与审稿风险预判)
- [十、下一步研究与顶会投稿路线图](#十下一步研究与顶会投稿路线图)
- [十一、代码目录与一键可复现资产全索引](#十一代码目录与一键可复现资产全索引)

---

## 零、执行摘要与全局进度总览（Go / No-Go 全绿通关）

**核心定位**：`KSSM`（Kernel State-Space Storage Engine）是首个**无需修改 Linux 内核源码、无需 Intel AMX/GPU 等专用矩阵硬件、无需改动任何上层应用代码，原生运行在 Linux `5.15` `eBPF` 验证器约束下的 $O(1)$ 内存、纳秒级（`37.8 – 53.3 ns`）双实体耦合选择性状态空间存储引擎**。

截至 2026 年 11 月 24 日，我们已完成从底层内核能力探针、核内微基准、跨应用零样本建模、审稿人最锋利消融（`Cross-Entity EWMA + 3L-MLP`）、读路径端到端混部（含锁频与强规则 `Heuristic_SeqDrop` 对抗）、写路径端到端 NVMe FDP 仿真与真机 `F_SET_RW_HINT` 闭环、10 类异构应用语料扩容、到统一生产引擎及 1–64 核并发扩展性压测的**全部 8 组系统性实证研究**：

| 实验编号与目录 | 核心检验目标 | 关键实测硬数据（未修改 Linux `5.15.0-177-generic`） | 状态 |
|---|---|---|---|
| **Exp 0-A**<br>`exp0a/` | 5.15 Verifier、零浮点 JIT、单次更新 `<60ns`、4KB CPU `<2%` | • 5/5 钩子一次过 Verifier（仅 `57 – 2,005 insns`, `<0.2%` of 1M limit），`objdump` 反汇编 `1,451` 条 x86_64 JIT 指令证实 **0 条 FPU/SSE/AVX/AMX 指令**<br>• 标量版 `1,003 ns` ❌ → **Contractive Guard-Bit SWAR 64D** 压缩至 **`45.94 – 58.14 ns`** ✅<br>• 朴素步长门控随机写 `6.88–8.57%` ❌ → **Mode-Transition + 128 KB Byte-Budget Gating** 压低 4KB 触盘 CPU 至 **`0.52% – 1.18%`**（`1.53M IOPS` 纯内存热读净增 `1.59% – 1.67%`） ✅ | 🟢 **PASS** |
| **Exp 0-B**<br>`exp0b/` | 5 类真实应用 LOAO 零样本第 0 秒寿命预测、消除 `eFDP` 失明期、`One-Shot Scan` AUC `>0.92` | • 5 类真实应用（`RocksDB/SQLite/PyTorch/LLM/ETL`）7 钩子零丢包实采<br>• Bit-exact `46 ns` `Guard-SWAR-Subspace`：第 0 秒寿命 BalAcc **`51.72%`**（领先 `PCStream 33.33%` 与 `Heimdall Faithful 33.33%` **`+18.39%`**），`+SlowLoop` 达 **`64.48% – 78.32%`**；切换窗 `0–10s` 即 `55.25%` vs `eFDP 33.33%`；`One-Shot Scan` 宏平均 AUC = **`0.9329`** ✅ | 🟢 **PASS** |
| **Exp 0-B+**<br>`exp0b_selective/` | 审稿人终极反问：正面硬刚 `Cross-Entity EWMA + 3L-MLP`；验证真·选择性门控 `<60ns` | • 提出 **4-Band Independent Selective Guard-SWAR 64D**（编译期立即数 4 路分派，消除 `rcx` 瓶颈）：Verifier `1,493 insns`，延迟 **`37.83 ns` (混合) / `50.19 ns` (最坏)**，噪声 `HOLD` 路径仅 **`3.47 ns`**<br>• 原 5 应用 LOAO：`True Selective KSSM`（单层线性头）BalAcc `58.05%` 击败最强 3 层浮点 `Multi-EWMA + 3L-MLP` (`53.01%`, `+5.04%`)<br>• 555K 真实内核多阶段交错噪声（同目录/同大小/同标志/同线程）：OOD 重噪声 **`K=35`** 下全部 3 种 `EWMA + 3L-MLP`（含 5 尺度 $\alpha=0.96875$）集体崩盘至 **`34.0%–34.6%`**，而 `True Selective Float64D` 稳守 **`62.58%`**，`4-Band Selective SWAR + Hard-STE QAT` (`37.8–50.2 ns`) 更是以 **`70.80%`**（总体 `72.35%`）**反超连续浮点上界 `+8.22%` 并碾压 EWMA `+36.3%`** ✅ | 🟢 **PASS** |
| **Exp 0-C**<br>`exp0c/` | 读路径端到端混部（`8 GiB cgroup v2`）：前台 `RocksDB` P99 降 `>=40%` 且后台 LLM 加载不掉速 | • 加载真实 `24.25 KB` 量化表，零文件名/PID 白名单，独立冷起跑 `file=5.11 GiB, pgscan=0`<br>• **纯 LLM 混部（100% 神经读出）**：RocksDB P99 `127 → 10 µs` (**`-92.1%`**)、QPS `3.14×`、后台 LLM 加载 `979.6 → 1,608.6 MB/s` (**`+64.2%`**)、`pgscan` 降 **`99.90%`**<br>• **全混部（LLM + 周期 Checkpoint）**：RocksDB P99 `163 → 7 µs` (**`-95.7%`**)、后台 LLM `1,012.6 → 1,914.2 MB/s` (**`+89.0%`**)、`pgscan = 0`；透明审计 **`RocksDB .sst` 误杀 = `0 files (0.00 GiB)`** ✅ | 🟢 **PASS** |
| **Exp 1**<br>`exp1_write_fdp/` | 写路径端到端实测：真机 `F_SET_RW_HINT` + 零 BPTT 慢回路 + 50 分钟动态切换 NVMe FDP WAF 与尾延迟 | • 硬件只读探测确认 PM174X 支持 NVMe Streams Directives 与 OCP `Physical media units written`；真机 Linux 5.15 实测 **`F_SET_RW_HINT` 40/40 写入内核验证通过** + 每 2 秒 `784 B` 零 BPTT 慢回路原子刷回内核（`on_vfs_open` 均摊 `87.78 ns`）<br>• 50 分钟（`3,000s`, `149.82 GiB`, 95% 峰值满载）跨应用切换 NVMe FDP 4-RUH 仿真（OP=`7%/14%/28%`）：标准企业盘 `OP=14%` 下 **`KSSM-Full (+SlowLoop)`** 将 WAF 压至 **`1.060`**、GC 搬迁量较 `No-Stream` 降 **`-75.9%`**、较强分箱 `eFDP+ (420s)` 降 **`-63.1%`**、P99 写尾延迟 **`932 µs`**（追平 `Oracle 916 µs`，较 `eFDP+ 420s 5,386 µs` 暴降 **`-82.7%`**） ✅ | 🟢 **PASS** |
| **Exp 2**<br>`exp2_scale_distill/` | 攻克量化与小样本短板：8-Band/4-Band Hard-STE QAT + 扩充至 10 类真实异构应用 10-Fold LOAO | • 补采 5 类异构真机负载（`g++` 编译链、`git`、C++ 日志段、`zstd` 归档、GPU 3 `vLLM KV Swap`，全部 0 丢包），构成 **`10` 类应用 / `569,008` 条真实内核事件语料库**<br>• 显著 Scaling Law：在原 5 应用上训练从 `4→9` 类后零样本 BalAcc 暴涨 **`+15.7% – +21.0%`**（至 `68.0%–73.8%`）；10-App 消融清晰拉开 `InodeOnly (63.1%) < Multi-EWMA+3L-MLP (64.1%) < FixedSWAR (65.2%) < Uncoupled (68.3%) < TrueSelective (69.0%) < +SlowLoop (73.8%, Acc 87.3%)`<br>• 回灌 Exp 1 FDP 仿真：仅凭 **9-App 纯冻结 `KSSM-ZeroShot`（零慢回路！）** 即在 `OP=7%` 切换窗 `0–420s` WAF (`1.241`) 与 P99 (`6,646 µs`) 双双反超带在线死亡学习的 `eFDP+ (420s)` (`1.249` / `7,076 µs`) ✅ | 🟢 **PASS** |
| **Exp 3**<br>`kssm_engine/` | 整合成单一生产级代码库 + 修复 `g_pair` OOV + `1 → 64` 核并发扩展性压测 | • 精简门控表为 `128 B` `g_s1[32][4]`，彻底修复 `log_segment` (`+46.5% Acc`) 与 `rocksdb` (`+33.7% Acc`) 两折 OOV，10-App 零样本 BalAcc 升至 **`67.74%`**，并自然涌现 `OPEN_CREAT/RENAME → PHASE_RESET` 状态机<br>• 统一 6 钩子引擎一次过 Verifier，真实 10-App 权重延迟 **`53.29 ns`**；提出 **`TL-PerCPU` 线程私有快路径门控**：64 核独立文件读 `3.02 MIOPS` **`100.1%` 零损失**；64 核死磕单共享热文件较 `SharedInode` 提速 **`+28.9%`**、均摊延迟砍半（`1,997 → 942 ns`）；32 核 NVMe 随机写 `0.50 MIOPS` **`100%+` 保持**（`BPF_CPU = 2.02%`） ✅ | 🟢 **PASS** |
| **Exp 4**<br>`exp4_read_hardened/` | 读路径锁频三元对抗压测：硬刚审稿人最强读规则 `Heuristic_SeqDrop` | • 锁定 CPU `0..15` `performance` governor + `8 GiB cgroup v2`；三元混部 `FG-1 RocksDB 5.2G` + `FG-2 HotSeqEmbed 512M` + `BG LLM 43G`<br>• `Heuristic_SeqDrop` 误杀 `79.19 GiB` 热顺序表导致 `FG-2` 吞吐腰斩至 `2,412 MB/s`（P99 `201.5 ms`）；**`KSSM Adaptive`**（64D SSM + 全局 1:1 `8 GiB` 2 MiB 簇重用自监督反馈，核内 `98.14 ns`）仅在第 1 遍试探降级 `0.48 GiB` (`0.21%`)，随后 454 遍 100% 驻留 DRAM：**`FG-1` P99 `-93.1%` (`116→8 µs`)、`FG-2` `6,652 MB/s`（`2.76×` vs `SeqDrop`, P99 `-62.7%`）、`BG LLM` 冲上全场最高 `2,754 MB/s`（`+53.5%` vs LRU / `+9.1%` vs `SeqDrop`）、`pgscan = 0`、`RocksDB .sst` 误杀 `0.00 GiB`** ✅ | 🟢 **PASS** |

---

## 一、研究背景、文献版图与核心灵感

### 1.1 为什么现代存储栈依然深陷两大顽疾？

在云数据中心与 AI 基础设施（训练/推理集群、多租户 KV/关系数据库混部节点）中，Linux 操作系统存储栈同时面临两条路径上的根本性语义鸿沟：

1. **读路径（Page Cache 缓存污染与预取失准）**：
   - 后台流式大读取（如加载数十 GB 的大模型 `.safetensors` 权重、数据库后台 Compaction 扫描、数据湖 ETL 顺序扫描）与前台延迟敏感热工作集（如 RocksDB/SQLite 随机点查、高频重用的共享 Embedding 表）共享有限内存（如 `cgroup v2` 容器配额）。
   - Linux 默认两链表 LRU（或启发式静态规则）无法在线准确分辨三类流：**① 随机热读**、**② 可驻留内存的高频重用顺序读**、**③ 扫完即弃的一次性冷扫描（`One-Shot Scan`）**。一旦冷流冲刷热页缓存，前台点查延迟立刻从 DRAM 级（`~4 µs`）劣化 30× 至 NVMe 磁盘缺页级（`~120–300 µs`），同时触发海量同步 Direct Reclaim 拖垮后台加载吞吐。
2. **写路径（冷热混写、写放大 WAF 与负载切换失明期）**：
   - 现代企业级 NVMe SSD 已经提供 Multi-Stream Directives 与 NVMe 2.0 Flexible Data Placement（FDP / Reclaim Unit Handles），允许主机将寿命相近的数据写入同一物理擦除块（Reclaim Unit, RU），以消除垃圾回收（GC）搬迁带来的写放大（WAF）与写尾延迟抖动。
   - 然而，绝大多数应用不会手动为每个文件标注寿命；而现有内核自动分流机制全部属于**响应式死亡追踪（Reactive Death Tracking）**——必须等一批文件真正被 `unlink` 或覆写死亡后才统计出其历史寿命。面对云上频繁的负载启动与多应用切换，响应式系统存在长达**数分钟的失明期**（如 `eFDP` 约 7 分钟重规划滞后），且小 I/O 追踪 CPU 开销高昂。

### 1.2 2019–2026 年 OS-ML 存储文献全景（严格区分正刊 / Workshop / arXiv）

并非“以前往 OS 内核塞的 ML 模型完全不看历史”，而是过去全部工作各自撞死在**「执行速度 vs. 长程选择性记忆 vs. 跨应用零样本泛化」**的结构断层里：

| 代表工作 | 刊物层级 | 模型骨架与历史处理机制 | 无法突破的系统硬墙 |
|---|---|---|---|
| **`LinnOS`** (OSDI '20)<br>**`Baleen`** (FAST '24)<br>**`Heimdall`** (EuroSys '25)<br>**`KML`** (ACM TOS '23) | **系统顶会 / 顶刊正刊** | 2–3 层浅层表格 MLP 或离线 GBDT；仅看当前孤立 I/O 标量或极短设备级滑窗（如 `LinnOS` 最近 4 次已完成 I/O 延迟与队列深度）；`KML` 原文将 RNN/LSTM 明确列为未实现的 Future Work。 | ① 极度短视且多进程混叠，无法跨越几十至数百次 syscall 维护应用阶段上下文；<br>② 绑定特定硬件延迟分布与单应用特征，**“一负载一重训”**，遇到新应用或负载切换即失效（HotOS '25 专门指出这一痛点）。 |
| **`Kleio`** (SYSTOR '19)<br>**`LAKE`** (ASPLOS '23) | **系统顶会正刊** | 真正具备长程循环记忆的 2 层 128 维 LSTM（`Kleio` 追踪每页过去 16 周期访问历史）；`LAKE` 通过内核↔共享内存↔用户态守护进程 RPC 卸载至 GPU CUDA 执行。 | 每步需要 $O(D^2)$ 稠密浮点矩阵乘，**根本进不了内核热路径**：离路径 GPU RPC 延迟高达几十微秒且抢占用户宝贵 GPU 算力，完全无法挂在每 I/O 的 VFS/块层关键路径上。 |
| **`PCStream`** (FAST '19)<br>**`BASIC-Prefetcher`** (SYSTOR '26) | **系统顶会正刊** | `PCStream` 维护静态调用栈 PC 哈希→历史平均死亡寿命映射表；`BASIC-Prefetcher` 按 I/O 大小分桶构建一阶马尔可夫转移链。 | ① 无上下文状态机：同一个库函数（如 `open()` / `torch.save()`）在不同阶段写出短命/长命截然不同的文件，静态 PC 哈希彻底失灵；<br>② 跨二进制或重编译后 PC 地址变化，命中率瞬间归零。 |
| **`eBPFML`** (eBPF '25) | **Workshop** | 探索在 eBPF 环境中运行神经网络。 | 认为运行稍复杂网络必须扩展 eBPF 指令集并依赖 Intel AMX 等专用矩阵硬件，无法在现役未修改 Linux 内核部署。 |
| **`eFDP`** (`arXiv:2611.06461`, 2026-11) | **arXiv 预印本** | 基于响应式块/文件死亡时间直方图与周期性流重规划驱动 NVMe FDP。 | ① 响应式冷启动盲区：负载启动或切换后存在**约 7 分钟（420 秒）重规划失明滞后**；<br>② 4KB 小写场景 CPU 开销高达 **8–9%**，制约高 IOPS 生产部署。 |
| **`Foreactor`** (`arXiv:2409.01580`)<br>**`PredictIO`** (`arXiv:2610.12971`) | **arXiv 预印本** | 通过应用级系统调用依赖图或专用插件预测 I/O。 | 需要程序员为每个应用手写依赖图或侵入式适配插件，违背 OS 透明性原则。 |

### 1.3 灵感起源与“三位一体”系统级突破（彻底打破“Model Swap”质疑）

在构思课题初期，我们面对一个极其尖锐的顶会审稿红线：**如果论文的核心叙事只是“把 MLP/LSTM 换成时髦的 SSM（Model Swap）”，OSDI / SOSP / FAST 审稿人会直接秒拒。** 真正把 `KSSM` 和上述全部工作拉开代差的，是紧密耦合的**三位一体系统级突破**：

1. **执行范式突破（填平 `Heimdall` 与 `LAKE` 之间的千年断层）**：
   - 以前 OS-ML 是一道二选一：要么 `LinnOS/Heimdall`（纳秒级进内核，但无状态短视），要么 `Kleio+LAKE`（有序列记忆，但拖着 GPU 走几十微秒离路径 RPC）。
   - 我们捕捉到**查表化对角选择性状态空间模型（Lookup-Table Diagonal Selective SSM）**与 Linux `eBPF` 验证器（无浮点、512 B 栈、10 寄存器、严格静态指令上限）的底层硬件同构性：每步严格 $O(D)$ 逐通道整数操作，零矩阵乘、零循环、零 FPU/AVX/AMX 污染。结合我们原创的 **Contractive Guard-Bit SWAR 4-Lane 打包**与 **4-Band 编译期立即数选择性门控**，首次在未修改 Linux 5.15 内核热路径上以 **`37.8 – 53.3 ns`** 跑通真·选择性序列模型（较 `LAKE` 提速近 3 个数量级）。
2. **状态抽象突破（跨实体因果传递：`task_struct` $\leftrightarrow$ `inode` 双态耦合）**：
   - 一个刚刚调用 `open(O_CREAT)` 诞生的新文件，其自身访问历史长度为 **`0`**！任何单客体序列模型（如单页 LSTM `Kleio` 或单文件死亡追踪 `eFDP`）在文件出生第 0 字节都是绝对瞎子。
   - `KSSM` 首次把隐状态绑定到操作系统最核心的一对因果对象——**执行主体（线程 `task_struct`）与数据客体（文件 `inode`）**，在每次 VFS 交互时双向门控交换状态。线程刚刚读完深层 SST 还是刚跑完验证集评估的上下文，在新文件 `open(O_CREAT)` 第 0 字节瞬间注入 $h^{\text{inode}}$，破局“第 0 秒预测”难题。
3. **泛化与抗噪突破（相对尺度不变行为语法 + 硬件精确零泄漏 `HOLD` + 零 BPTT 慢回路）**：
   - 彻底剥离裸 PC 地址、物理 LBA 和绝对目录周转率量纲，用 5 元相对行为语法实现跨二进制零样本泛化（`24.7 KB` 底座常驻 L1d Cache）；
   - 通过输入相关选择性遗忘门打破固定衰减 EWMA 的 **Stability-Plasticity 两难**（在多阶段重噪声下实现零衰减泄漏 `HOLD`，且仅耗时 `3.47 ns`）；
   - 配合延迟死亡/重用真值触发的**零 BPTT 慢回路**（用户态每 2 秒原子热更新 `< 1.1 KB` 线性读出头），兼顾第 0 秒零样本冷启动与稳态 `90%+` 精度。

---

## 二、最终系统愿景、目标与研究意义

### 2.1 最终目标蓝图
打造一个**零内核补丁、零硬件依赖、零应用改动、双任务共享底座的 Linux 原生 `eBPF` 存储智能引擎（`KSSM`）**：
- **统一底座**：每个活跃线程与活跃文件各维护固定 `128 Bytes`（64 维定点）隐状态，一套常驻 L1d Cache 的 `24.7 KB` 查表化选择性对角 SSM 底座实时推演状态；
- **读/缓存控制面**：实时识别 `One-Shot Scan` vs `Hot Reusable`（包括随机热读与可重用热顺序读），经 Ringbuf 异步驱动页缓存 `drop-behind`（保留 `16 MB` 预读安全垫），保护前台低延迟服务并释放 NVMe 带宽；
- **写/分流控制面**：在文件出生第 0 字节预测寿命档位（Short / Medium / Long），通过 Linux 原生 `F_SET_RW_HINT` / NVMe Multi-Stream / FDP RUH 实施冷热隔离，彻底消除负载切换 7 分钟失明期与高写放大；
- **自监督慢回路**：内核在真实发生 `unlink`/覆写或页簇二次命中时回传延迟真值，用户态守护进程无需反向传播（Zero-BPTT），仅用轻量在线逻辑回归每隔 2 秒原子更新内核中 `1,040 B` 的双任务线性读出头。

### 2.2 核心研究与工程意义
- **学术意义**：证明了在严格静态验证沙箱（未修改 Linux `eBPF` Verifier）中，通过「收缩映射数学约束 + 64-bit SWAR 寄存器内并行 + 操作系统双实体状态抽象」的深度协同设计，可以在纳秒尺度原生运行具备输入相关选择性记忆的序列神经网络，填补了 OS-ML 领域 `Heimdall` 与 `LAKE` 之间的空白。
- **工业部署价值**：无需重编译内核、无需开启易冲突的全局 boot 参数、无需昂贵 AMX/GPU 硬件、极端 4KB 写 CPU 开销 `< 2%`、64 核并发零损失扩展，可直接热加载至现役云数据中心与 AI 推理/训练集群节点。

---

## 三、KSSM 模型与系统架构深度设计（数学推导 + 字节级内核实现）

### 3.1 完整数据流架构

```
  [VFS / Ext4 内核热路径: ext4_file_{read,write}_iter, vfs_open, vfs_unlink, vfs_fsync_range]
                                         │
          ┌──────────────────────────────┴──────────────────────────────┐
          ▼ (快路径 ~96.5% 4KB I/O 跳过)                                ▼ (慢路径 ~3.5%: 元数据 / 模式拐点 / 128KB 字节预算)
  [Mode-Transition + 128KB Gating]                              [5 元相对行为语法提取 u_t = (s1, s2, s3, s4, s5)]
   (TL-PerCPU: 仅读写私有 TASK_STORAGE                           • 零裸 PC, 零裸 LBA, 按线程步长, 相对尺度周转率 s5
    or 按 inode 字节预算, 仅 ~40-85 ns)                                  │
                                                                         ▼
                                                       [4-Band Selective Guard-SWAR 64D]
                                                        • 查 24 KB emb[192][16] + 128 B reg_s1[32][4] (常驻 L1d)
                                                        • 4 个 16 维子带独立择一: HOLD(3.5ns) / SLOW / FAST / RESET(28ns)
                                                        • 同步双向耦合更新 h_inode (128B) <-> h_task (128B)
                                                        • 严格 Banach 收缩不变集 [8192, 24575], 0 溢出, 0 浮点, 37.8-53.3 ns
                                                                         │
                              ┌──────────────────────────────────────────┴─────────────────────────────────────────┐
                              ▼                                                                                    ▼
               [读/缓存头: w_scan (260 B)]                                                          [写/寿命头: w_life (780 B)]
               + 1:1 全局 8 GiB 2MiB 簇重用自监督表                                                 • 出生第 0 字节预测 Short/Med/Long
               • 精准三分类: 随机热 / 热顺序 / 一次性冷扫                                           • 死亡 vfs_unlink 回传出生快照与真值
                              │                                                                                    │
                              └───────────────────────────► [BPF_MAP_TYPE_RINGBUF (8 MB)] ◄────────────────────────┘
                                                                         │
                                                                         ▼ (异步低频, 0 丢包)
                                                       [用户态统一控制守护进程 kssm_daemon]
                                                        • DROP_HINT  -> posix_fadvise(DONTNEED) / sync_file_range (16MB 安全垫)
                                                        • BIRTH_HINT -> fcntl(F_SET_RW_HINT) 写入内核 inode->i_write_hint
                                                        • DEATH_FB   -> 每 2s 零 BPTT 在线逻辑回归原子刷回 1,040 B head_map
```

### 3.2 双实体隐状态与 Linux 5.15 内核存储适配

每个活跃线程与每个活跃常规文件各自绑定一个固定 **`128 Bytes`** 的隐状态向量：
- 逻辑维度 $D = 64$，每个通道用 `16-bit` 定点数表示（`64 × 2 Bytes = 128 Bytes`），在内存与寄存器中按 **`uint64_t h[16]`**（每 64 位寄存器打包 4 个 16-bit 通道）紧凑排布。
- **线程隐状态 `h_task`**：存于 Linux 5.15 原生 `BPF_MAP_TYPE_TASK_STORAGE`（`BPF_F_NO_PREALLOC`，内核挂在 `task_struct->bpf_storage` 上，完全线程私有、零锁、L1d 独占）。同时在同一结构体中存放 16 槽最近调用栈环 `stack_ring[16]`、近期读写计数、线程步长追踪字段 `last_ino/last_off/last_cnt/seq_run` 及 `TL-PerCPU` 快路径计数器 `bytes_acc`。
- **文件隐状态 `h_inode`**：
  - 在支持 `BPF LSM`（开启 `lsm=...,bpf` 启动参数）或 Linux `6.4+` 环境下，直接存于原生 `BPF_MAP_TYPE_INODE_STORAGE`（我们已用 `lsm_kssm_unified_native` 实测通过 Linux 5.15 Verifier）；
  - 在未修改启动参数的现役 Linux `5.15` 生产内核上（实测发现 `lsm/file_open` 静默不触发、且 `fentry` 禁调 `bpf_inode_storage_get`），无缝使用 `BPF_MAP_TYPE_LRU_HASH | BPF_F_NO_COMMON_LRU`（Per-CPU 独立 LRU 自由链表，消除全局 LRU 锁争用），key 为 64 位 `inode->i_ino`。

### 3.3 去绝对量纲的 5 元相对行为语法 $u_t = (s_1, s_2, s_3, s_4, s_5)$

**铁律：绝不向模型输入裸用户态 PC 地址、物理磁盘 LBA、文件名/路径字符串、PID 或无量纲绝对 IOPS**。5 个离散槽位合计 $32 + 64 + 32 + 32 + 32 = 192$ 槽：

| 槽位 | 词表范围 | 物理语义与在线严格因果计算方式 | 跨应用零样本泛化价值 |
|---|---|---|---|
| **$s_1$** | `[0, 31]` (32 槽) | **VFS 原语与访问模式**：`0=OPEN_RO`, `1..4=OPEN_CREAT(+O_APPEND/+O_TRUNC)`, `5..6=OPEN_RW(+O_APPEND)`, `7=TRUNC`, `8=SEQ_READ`, `9=RND_READ`, `10=WRITE_APPEND`, `11=SEQ_OVERWRITE`, `12=RND_OVERWRITE`, `13=FSYNC`, `14=UNLINK`, `15=RENAME`, `16=MMAP_SEQ`, `17=MMAP_RND`。 | 操作系统通用原语语言，与上层编程语言和二进制完全无关。 |
| **$s_2$** | `[0, 63]` (64 槽) | **相对调用栈重现签名 + 线程读写比**：用 `bpf_get_stackid(BPF_F_USER_STACK \| BPF_F_FAST_STACK_CMP)` 获取栈 ID（失败回退内核栈），在 `task_storage` 的 16 槽最近调用栈环中做**纯位运算无分支扫描**求相对重现距离 `dist ∈ [0..15]`（15=首次出现，0=紧邻重复），高 2 位拼接该线程近期写比例四分位桶 `rw_q ∈ [0..3]`：`s2 = (rw_q << 4) \| dist`。 | 彻底替换绝对 PC 地址：捕获“紧密循环扫描 (`dist=0..2`) vs 阶段切换新函数 (`dist=15`)”与“读线程/写线程/混合 Compaction 线程”角色。 |
| **$s_3$** | `[0, 31]` (32 槽) | **文件内归一化几何位点**：高 3 位为文件当前大小对数档位 `fast_sz_b(i_size) ∈ [0..7]`（`<4K, <64K, <1M, <16M, <256M, <1G, <4G, >=4G`），低 2 位为当前偏移所在几何区域 `region ∈ {0=Header(<64KB), 1=前半部, 2=后半部, 3=Tail/Append}`。 | 捕获文件头元数据读取、大文件尾部追加、随机中段查表等几何语义。 |
| **$s_4$** | `[0, 31]` (32 槽) | **请求大小 + 按线程追踪步长/持续顺序度**：高 2 位为请求大小档位 `req_b ∈ [0..3]`（`<16K, <64K, <256K, >=256K`），低 3 位为基于**线程私有 `last_off`**（防多线程共读同一 SST 互相污染！）计算的步长桶：`0=持续顺序(seq_run>=3)`, `1=短回跳(>-128K)`, `2=长回跳`, `3=同偏移重读(stride=0)`, `4=单步顺序`, `5..7=短/中/长正向跳读`。 | 区分持续流式扫描 vs 局部回绕重读 vs 随机点查。 |
| **$s_5$** | `[0, 31]` (32 槽) | **尺度不变相对父目录瞬态周转率**：内核 `dir_map` 追踪父目录近期 `create/unlink/rename` 指数衰减计数 `ch`；进程维护因果滑动均值 `ema_ch`，计算相对差值 `rel_delta = log2(ch+1) - ema_ch` 映射至低 4 位 `[0..15]`，高位 `bit 4` 标示目录是否静默 (`ch <= 4`)。 | 解决跨应用周转率 100× 量纲偏移（如 RocksDB 绝对周转 9.3 vs LLM 绝对周转 5.6），跨应用完美对齐相对冷热目录语义。 |

### 3.4 数学核心：Contractive Guard-Bit SWAR 与 4-Band 选择性门控推导

#### （1）为什么原始标量公式物理上不可能 `< 60 ns`？
在 `storage-192` 的 `2.10 GHz` Xeon 6530 上，**`60 ns` 仅等于 `126 个时钟周期`**。`perf stat` 实测原标量 `Exact Q7.8 64D` 虽已达到 **`5.46 IPC`**、L1d miss 仅 `0.17%`，但 64 通道循环 + 10 路查表 + `4×64` 标量 `imul` + 寄存器 spill 总计执行约 **`11,500` 条 x86_64 指令**，耗时高达 **`1,002.89 ns`**，超过物理天花板 16.7 倍。

#### （2）Banach 收缩不变集 + Excess-16384 Guard-Bit SWAR 数学证明
为在纯 64-bit 标量 eBPF ALU 指令（0 FPU / 0 SSE / 0 AVX / 0 AMX）下实现 4 路并行且彻底消灭跨 lane 进位与饱和裁剪分支，我们构造如下编码：
- **隐状态编码（Q5.8 Excess-16384）**：将每个通道的逻辑有符号值 $h \in [-8192, +8191]$（实数范围 $[-32.0, +32.0)$，精度 $2^{-8} = 0.0039$）存储为 16-bit 无符号整数 $H = h + 16384 \in [8192, 24575]$（`0x2000 .. 0x5FFF`）。**每个 16-bit lane 的最高位 bit15 恒为 0**，距上溢 `0xFFFF` 有 `40,960` 保护裕度，距下溢 `0x0000` 有 `8,192` 保护裕度。每 4 个 lane 打包为 1 个 `uint64_t`，64 维隐状态仅需 **`16 × uint64_t`（`128 Bytes`）**。
- **查表偏置编码（Excess-2048）**：离线预烘表项 $e_k$ 存为无符号 $E_k = e_k + 2048 \in [1536, 2559]$。无论 2 槽求和（`Subspace`: $B = E_a + E_b = e_a + e_b + 4096$）还是 5 槽求和（`Shared`: $B = \sum_{k=1}^5 E_k = \sum e_k + 10240$），直接用 **1–4 条原生 64-bit `add`** 同时完成 4 路加法，绝不产生跨 lane 进位！
- **多尺度移位衰减与双态耦合**：对于衰减指数 $d \in \{1,2,3,4\}$（$\alpha = 1 - 2^{-d}$）与耦合指数 $c > d$（$\gamma = \pm 2^{-c}$），每 4 通道并行的无分支精确整数更新为：
  $$H_i' = H_i - \bigl((H_i \gg d) \,\&\, M_d\bigr) + \bigl((H_t \gg c) \,\&\, M_c\bigr) + B - C^{+}_{d,c}$$
  $$H_t' = H_t - \bigl((H_t \gg d) \,\&\, M_d\bigr) - \bigl((H_i \gg c) \,\&\, M_c\bigr) + B - C^{-}_{d,c}$$
  其中 $M_k = \operatorname{PACK4}(\text{0xFFFF} \gg k)$，精确偏置校正编译期常量为：
  $$C^{+}_{d,c} = \operatorname{PACK4}\!\left(B_{\text{off}} + \frac{16384}{2^c} - \frac{16384}{2^d}\right), \quad C^{-}_{d,c} = \operatorname{PACK4}\!\left(B_{\text{off}} - \frac{16384}{2^c} - \frac{16384}{2^d}\right)$$
- **不变集严格安全性定理（Banach Contraction Invariant Set）**：
  由于 $c > d \implies \rho = 1 - 2^{-d} + 2^{-c} < 1$，只要离线量化时约束总输入偏置幅度满足
  $$|b_{\max}| \le (2^{-d} - 2^{-c}) \cdot 8192$$
  则归纳法立刻保证：若 $|h_{t-1}| \le 8192$，必有
  $$|h_t| \le (1 - 2^{-d})|h_{t-1}| + 2^{-c}|h^{\text{other}}_{t-1}| + |b_{\max}| \le (1 - 2^{-d} + 2^{-c})\cdot 8192 + (2^{-d} - 2^{-c})\cdot 8192 = 8192$$
  **推论**：$\forall t \ge 0,\ H_t \in [8192, 24575]$ 恒成立，**零溢出、零跨 lane 进位、彻底免去饱和截断指令**！（实测连续 `370 万` 步压测及 10 类真实应用全量流式推理中，所有隐状态始终收敛在 `[11571, 19529]` / `[14958, 17903]` 安全腹地内。）

#### （3）4-Band Independent Input-Dependent Selective Guard-SWAR（消除 `rcx` 瓶颈）
- **为什么固定移位不够？** 若 $(d, c)$ 恒定不变，模型在数学上退化为多尺度固定衰减 EWMA，在多阶段交错重噪声下会崩盘。
- **为什么运行时动态移位 `>> g->d_i` 不行？** 实测发现 x86_64 可变移位指令 `shr rax, cl` 强迫每次移位串行争抢单寄存器 `rcx`，导致延迟翻倍至 `87.35 ns`（超 60 ns 红线）。
- **最终突破（编译期立即数 4-Regime 独立子带分派）**：将 64 维隐状态划分为 4 个 16 维子带（每带 `4 × uint64_t`），子空间因子化输入分配为 `Band 0 ← s1+s2`、`Band 1 ← s1+s3`、`Band 2 ← s1+s4`、`Band 3 ← s1+s5`。每条事件以 $s_1 \in [0..31]$ 查 `128 Bytes` 只读门控表 `reg_s1[s1][0..3]`，每个子带独立择一执行 4 种编译期立即数模式：
  - **`0 = HOLD`（噪声免疫跳过）**：$\alpha=1, \gamma=0, \bar{b}=0$ → **0 条算术指令直接跳过（全跳仅 `3.47 ns`）**，数学上零衰减泄漏！
  - **`1 = SLOW_DECAY`（长程记忆）**：$d=3, c=4$ ($\alpha = 0.875, \gamma = \pm 0.0625$)。
  - **`2 = FAST_DECAY`（快速适应）**：$d=1, c=2$ ($\alpha = 0.5, \gamma = \pm 0.25$)。
  - **`3 = PHASE_RESET`（阶段边界重置 + 强跨实体注入）**：$\alpha^{\text{inode}} = 0, \gamma^{\text{t}\to\text{i}} = 0.5, \alpha^{\text{task}} = 0.75$ → `vi[g] = ((ot >> 1) & 0x7FFF7FFF7FFF7FFF) + b + PACK4(4096)`，瞬间清空旧文件态并高增益继承 `h_task`（全重置仅 `28.31 ns`）！
- **字节级总足迹**：`kssm_base_model = 24,576 B (emb) + 128 B (reg_s1) = 24,704 B`（完整常驻 `48 KiB` L1d）；`kssm_dual_head = 1,040 B`（`w_scan` 260 B + `w_life` 780 B）。

### 3.5 双层高频节流门控与读路径 1:1 簇重用自监督反馈

1. **Mode-Transition + 128 KB Byte-Budget Gating**：
   - 仅在① 元数据事件（`open/unlink/rename/fsync`）、② 访问模式真实切换（`seqR ↔ rndR ↔ seqW ↔ rndW`）、③ 累计 I/O 达 `128 KB` 时才调用 `bpf_get_stackid()` 与 64D SSM 更新，将顺序与随机 4KB I/O 的最坏稳态触发率均封顶在 `~3.2% – 3.7%`。
   - **线程私有 vs 文件共享快路径的精准适用边界**（由 Exp 3 与 Exp 4 双实测得出）：
     - 多核共享少数热点大文件高并发场景 → 启用 **`TL-PerCPU`**（快路径 `bytes_acc` 放线程私有 `TASK_STORAGE`，消除 64 核缓存行乒乓，提速 `+28.9%`）；
     - 单线程高速跨几十上百个 SST 文件随机点查场景（如 RocksDB）→ 启用按 `inode` 的 `128 KB` 字节预算（避免 `ts->last_ino != ino` 导致跨文件跳读击穿快路径）。
2. **读路径 1:1 全局 2 MiB 簇重用自监督表（Exp 4 突破）**：
   - 维护一个全局 `BPF_MAP_TYPE_LRU_HASH`（`4,096` 槽 × `2 MiB` 采样步长 = 精确 `8 GiB` 滑动窗口，仅 `128 KB` 内存，**1:1 匹配 `8 GiB` cgroup 物理内存驻留边界**），仅在文件偏移跨过 `2 MiB` 时探测 1 次（几乎 0 开销）。
   - 用 `alpha = 1/4` 的指数滑动均值追踪重用率 `reuse_ema`；当 `reuse_ema >= 64`（近期至少 25% 的 `2 MiB` 块在 `8 GiB` 缓存窗口内被二次重读）时，自监督校准自动抑制对该文件的 `One-Shot Scan` 冷降级，无需任何文件名规则即可精准保住 `512 MiB` 热顺序表而放行 `43 GB` 一次性大模型加载。

---

## 四、离线训练、量化感知微调（Hard-STE QAT）与零 BPTT 慢回路方法论

### 4.1 离线训练配置与双实体因果窗口构造

- **硬件与环境**：严格使用 `storage-192` 的 GPU 3（`CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3`，`NVIDIA A100-PCIE-40GB`，PyTorch `2.13.0+cu130`）。
- **因果训练窗口构造**：
  - 对每个标注样本（文件出生第 0 字节 `life_ev` 或读重用采样点 `scan_ev`），按目标线程 `tid` 与目标文件实例 `inst` 分别向过去严格回溯（不泄露任何未来事件！），合并排序截取最近 `WIN_LEN = 32` 步（多阶段噪声压测 `WIN_LEN = 48` 步）完整因果事件序列；
  - 批训练时维护 `mi`（是否命中目标文件）与 `mt`（是否命中目标线程）掩码（或局部 `K=10` 活跃 inode 槽位矩阵），在 GPU 3 上以 `batch_size = 512` 并行展开双实体耦合递推。
- **类均衡 + 跨应用等权联合多任务损失**：
  - 针对真实负载类别极度不平衡（如 RocksDB/SQLite 中 Short 占 86–95%），对每个训练应用内部按逆类频率加权并归一化，**确保每个训练应用贡献完全相等的梯度总权重**（防止样本量最大的 RocksDB 垄断底座表征）；
  - 联合优化读/写双任务头 + **1D Total-Variation 平滑正则**（强制有序邻近桶平滑外推，彻底消除低频桶 OOV 断层且 0 运行时开销）：
    $$\mathcal{L} = \mathcal{L}_{\text{CE}}^{\text{life}}(\text{balanced}) + \mathcal{L}_{\text{BCE}}^{\text{scan}}(\text{balanced}) + \lambda_{\text{TV}}\!\left(\sum_{s \in s_5}\|E[s+1]-E[s]\|_2^2 + \frac{1}{2}\sum_{s \in s_2}\|E[s+1]-E[s]\|_2^2\right)$$
  - 优化器：`AdamW`（`lr = 6e-3 ~ 8e-3`, `weight_decay = 2e-3`, `grad_clip = 2.0`），`CosineAnnealingLR` 调度 `32–40` epochs，读出前 `Dropout(0.15)`。

### 4.2 温度退火 + 末段 Hard-STE QAT（为什么优于连续门控蒸馏）

为使离线训练的 `4-Band Selective Guard-SWAR` 与 Linux `5.15` 内核整数代码 **100% 位级对齐（Bit-Exact）**，训练采用两阶段量化感知策略：
1. **阶段 A（前 75% epochs：指数温度退火软选择）**：每个频带的 4 档门控概率 $p_b = \operatorname{softmax}(g_{s_1, b} / \tau)$，$\tau$ 从 `2.0` 指数退火至 `0.06`，同时嵌入权重经 $\tanh(\cdot) \times 0.75$ 严格锁定在 Banach 收缩安全界内；
2. **阶段 B（最后 25% epochs：Hard-STE 离散微调）**：前向切为精确 `argmax` 硬独热选择（`oh = one_hot(argmax(p))`），反向经 Straight-Through Estimator（`probs = (oh - probs).detach() + probs`）传回梯度，彻底消除训练→部署的离散化休克（Discretization Shock）！
3. **关键消融教训（为何弃用 Naive MSE 门控蒸馏）**：我们在 Exp 2 中严格对比了 Teacher (`Float64D`) → Student (`4-Band/8-Band`) 的软 Logits KL + 隐状态余弦 + 门控 MSE 蒸馏，实测发现 `MSE` 连续门控蒸馏会把 Student 原本学出的精确硬 `HOLD (0.0)` 噪声跳过率从 `32.5%` 拉糊成 `22.8%`，导致 `K=35` 准确率从 **`70.80%`** 暴跌至 `36.95%`；而直接采用 **Hard-STE QAT** 不仅零成本、更简洁，还凭借硬件精确零泄漏 `HOLD` 反超连续浮点 Teacher **`+8.22%`**！

### 4.3 离线位级预烘与在线零 BPTT 慢回路校准

- **位级预烘导出**：
  - `emb_u16 = (round(tanh(emb) * 0.75 * 256.0) + 2048).astype(uint16)`，每 4 通道打包为一个小端 `uint64_t`，写入 `models/kssm_base.bin`（`24,576 B`）；
  - `reg_s1 = argmax(g_s1, dim=-1).astype(uint8)`，追加写入 `models/kssm_base.bin`（`128 B`，合计 `24,704 B`）；
  - 双读出头权重按 `Q7.8`（`× 256`）量化为 `int16_t`，偏置按 `Q15.16`（`× 65536`）量化为 `int32_t`，写入 `models/kssm_head.bin`（`1,040 B`）。
- **零 BPTT 慢回路在线校准**：
  - 内核 `kssm_vfs_open` 在新文件出生第 0 字节快照 `(snap_hi, snap_ht)`；当 `kssm_vfs_unlink` 真实死亡到达（或存活超过 `55s` 确认长命）时，用户态守护进程对冻结隐状态特征 $x \in \mathbb{R}^{128}$ 执行单步带类别加权的在线逻辑回归梯度更新：
    $$W_k \leftarrow W_k - \eta_y \bigl(\operatorname{softmax}(Wx+b)_k - \mathbb{I}[k=y]\bigr) x^\top$$
  - 每隔 **`2 秒`** 将校准后的 `1,040 B` `struct kssm_dual_head` 通过单次 `bpf_map_update_elem` 原子热刷回内核 `head_map`——**底座 `24.7 KB` 永久冻结、零反向传播、零 BPTT、用户态 CPU 占用 `< 0.05%`**。

---

## 五、实验平台勘察、数据采集体系与严格实验卫生

### 5.1 `storage-192` Linux `5.15.0-177-generic` 底层能力实测勘察

| 检查项 | 实测状态 | 对架构实现的直接约束与适配 |
|---|---|---|
| CPU / 缓存 / 内存 | 双路 Xeon Gold 6530 (128 核), 稳态 `2.10 GHz`, L1d `48 KiB`/核, L2 `2 MiB`/核, `503 GiB` DRAM | `60 ns = 126 cycles` 成为标量屋顶线；查找表 `24.7 KB` 完整装入 L1d；支持 `performance` / `ondemand` governor。 |
| BTF / 工具链 | `/sys/kernel/btf/vmlinux` (5.28 MB), `BPF_JIT_ALWAYS_ON=y`, `clang 18.1.3`, `libbpf 1.3.0`, `bpftool v7.4.0`, `perf 6.8.12` | CO-RE 与硬件 JIT 完整可用，`sudo -n` 免密。 |
| `TASK_STORAGE` | ✅ `fentry` / `lsm` / `kprobe` / `raw_tracepoint` 全部支持 | 原生承载 `h_task`、16 槽栈环与线程步长/快路径状态。 |
| `INODE_STORAGE` & `BPF LSM` | ⚠️ **重大实测发现**：① `fentry` 调 `bpf_inode_storage_get#145` 被 5.15 Verifier 拒收 (`-22`)；② `CONFIG_BPF_LSM=y` 但 `/sys/kernel/security/lsm` 仅含 `lockdown,capability,landlock,yama,apparmor`（无 `bpf`），导致 `SEC("lsm/file_open")` 可过 Verifier 且 `ATTACH OK` 但运行期 **`lsm_hits = 0`（静默空转）** | 双轨设计：用 `lsm_kssm_unified_native` 验证原生 `TASK_STORAGE + INODE_STORAGE` 过 5.15 Verifier；生产 `fentry` 采用 `Per-CPU LRU_HASH (BPF_F_NO_COMMON_LRU)` 零重启落地。 |
| `vfs_read` arg3 `loff_t *pos` | ⚠️ 5.15 Verifier 禁访基本类型指针参数 (`arg3 type INT is not a struct`) | 改挂 `fentry/ext4_file_{read,write}_iter(struct kiocb *iocb, ...)` 直接取 `iocb->ki_pos`，天然全覆盖 `read/write/pread/pwrite/readv/writev/io_uring`。 |
| NVMe 硬件 `/dev/nvme7n1` | Samsung PM174X (`MZWLO3T8HCLS-00B07`, 3.84 TB ext4 `/mnt/nvme12`), `oacs=0xff` | 只读探测确认原生支持 **NVMe Streams Directives (`type=1`)** 与 **OCP `Physical media units written`** 字节级硬件 WAF 计数器；不支持 NVMe 2.0 FDP opcode，且因多用户共享盘（已用 2.5 TB）严禁整盘 Preconditioning 破坏性测试，写路径采用「真机 `F_SET_RW_HINT` 闭环 + 高精度 C++ FDP FTL 仿真」双轨验证。 |

### 5.2 7 钩子零丢包内核采集器与 10 类真实异构应用语料库

编写 `exp0b/kssm_collector.bpf.c`（7 个 `fentry` 钩子：`ext4_file_read_iter`, `ext4_file_write_iter`, `vfs_open`, `vfs_unlink`, `vfs_rename`, `vfs_fsync_range`, `filemap_fault`，`32 MB` `BPF_MAP_TYPE_RINGBUF`），串联采集 **10 类完全不同的真实系统/生产存储应用**（共 `569,008` 条纯净内核事件）以及 **2 份多阶段交错噪声压测轨迹**（`939,263` 条事件），**全部 12 份真机轨迹 `ringbuf_drops = 0`（绝对零丢包），`user_stack_id` 有效率 `100%`**：

| # | 应用名 | 真实运行程序与二进制类型 | 纯净事件数 | 线程/子进程数 | 第 0 秒标注文件数 (`Short<10s / Med 10-50s / Long>=50s`) | 读重用标注事件 (`Hot<16MB / Scan>=64MB或永不重用`) | 读顺序率 (`Hot` vs `Scan`) |
|---|---|---|---:|---:|---|---|---|
| 1 | `rocksdb` | C++ `librocksdb 8.9.1`（220K keys ~220 MB, 多层 Compaction + 热键点查 + 全库扫描） | `410,887` | `8` | `1,576` (`1363 / 206 / 7`) | `365,751` (`307,949 / 57,802`) | `0.51` vs `0.65` |
| 2 | `sqlite` | `sqlite3 3.45.1`（DELETE journal + WAL 滚动分片 + 16×16MB 一次性全表扫） | `33,496` | `4` | `954` (`906 / 24 / 24`) | `22,293` (`20,905 / 1,388`) | `0.34` vs `0.68` |
| 3 | `pytorch` | PyTorch 2.13（GPU 3 训练 + staging / 滚动 `step_ckpt` / `best_ckpt` + 训练分片流） | `4,870` | `1` | `208` (`101 / 77 / 30`) | `3,001` (`18 / 2,983`) | `0.11` vs `0.95` |
| 4 | `llm` | 真实加载 `DeepSeek-R1-Distill-Qwen-14B` (28G) & `Llama-3-8B` (15G) `.safetensors` + KV offload | `32,731` | `3` | `106` (`68 / 26 / 12`) | `19,735` (`4,833 / 14,902`) | `0.13` vs `0.95` |
| 5 | `etl` | 外部归并 3 阶段（同体积 2 MB 产物）+ 32 MB 大文件顺序扫 + 热索引随机读 | `18,202` | `4` | `130` (`76 / 29 / 25`) | `14,570` (`1,105 / 13,465`) | `0.42` vs `0.97` |
| 6 | `compiler` | 真实 `/usr/bin/g++`, `cc1plus`, `as`, `ld` 多模块编译构建 + 热头文件 + 大 TU 扫描 | `24,606` | `445` | `418` (`294 / 83 / 41`) | `14,348` (`13,449 / 899`) | `0.44` vs `0.86` |
| 7 | `git_vcs` | 真实 `/usr/bin/git` 高频 `add/commit/gc` + 热 `.git/index` + 冷 pack 扫描 | `21,062` | `551` | `1,434` (`1335 / 46 / 53`) | `2,682` (`1,650 / 1,032`) | `0.28` vs `0.76` |
| 8 | `log_segment` | 原生 C++ 多线程 Append-Only Log/AOF 段滚动与后台段压缩引擎 | `9,324` | `4` | `230` (`138 / 54 / 38`) | `2,827` (`2,307 / 520`) | `0.28` vs `0.86` |
| 9 | `zstd_arch` | 真实 `/usr/bin/zstd` 增量/全量压缩归档流水线 + 热清单随机查 + 流式压缩 | `8,504` | `206` | `188` (`111 / 43 / 34`) | `5,054` (`2,472 / 2,582`) | `0.56` vs `0.99` |
| 10 | `vllm_kv` | GPU 3 `torch` fp16 张量 D2H KV-Cache 抢占换出 / 会话 Prefix / 持久 LoRA（统一 1 MB 大小） | `5,326` | `4` | `208` (`124 / 48 / 36`) | `2,320` (`1,806 / 514`) | `0.40` vs `1.00` |
| 11–12 | `mp_train` / `mp_test` | 多阶段交错噪声压力轨迹（同输出目录 / 同 512 KB / 同 flags / 同线程；Train `K∈{0,3,8}`, Test `K∈{0,5,15,35}`） | `383,655` / `555,608` | `1` | `35,200` / `33,867` 等权 3 类 | 专注测试选择性抗噪 | — |

### 5.3 严苛的实验卫生与防作弊协议

1. **Ext4 `inode` 号复用精确消歧**：追踪 `(ino, generation)`，每次 `EV_UNLINK` / `EV_TRUNC` 递增 `generation`，确保同一 `i_ino` 后续新建文件绝不串用旧寿命。
2. **严格 22 秒右截断剔除（Right-Censoring Guard）**：凡出生在采集结束前最后 `22 秒` 内且结束时未死亡的文件，因无法区分中命（16–17s 淘汰）与长命，一律剔除不参与训练和测试；且数据目录清理严格在 `kssm_collector` 停止之后执行，绝不把事后 `rm -rf` 误标为长命文件死亡。
3. **预测时刻绝对锁定第 0 字节 + 完整流式推理**：所有第 0 秒寿命评估严格取 `EV_OPEN_CREAT`（文件写入 0 字节瞬间）；测试阶段绝不用短窗截断，而是把量化后的 `uint16` 表放进与 eBPF C 代码**位级等价**的整数流式引擎，从未见应用的第 0 条事件连续演进到最后 1 条事件。
4. **黄金标准页重用真值（Belady-style 前向字节重用距离）**：对每个读窗口精确向后查找同一 `(inst, 128KB cluster)` 下次被读入前间隔的总读取字节数 $\Delta B$：$\Delta B < 16\text{ MiB} \implies \text{Hot Reusable (0)}$；$\Delta B \ge 64\text{ MiB}$ 或在去除末尾 15% 尾段后永不重用 $\implies \text{One-Shot Scan (1)}$。
5. **端到端系统实验卫生**：每组实验重建全新 `8 GiB cgroup v2`（`memory.swap.max=0`）、仅对测试目标文件局部调用 `posix_fadvise(DONTNEED)` 预清冷启动（绝不触碰全局 `drop_caches` 影响同机他人）、Exp 4 锁定测试核 CPU `0..15` 为 `performance` governor、守护进程**绝对不看文件名/目录/PID** 并在退出时打印逐文件真实路径误杀审计 `AUDIT_BREAKDOWN`。

---

## 六、全部对比方法与内部消融设计（确保苹果对苹果公平性）

为了杜绝“把基线做弱”的审稿风险，我们在各阶段共设计并实现了 **15 个外部基线与内部消融变体**，均在完全相同的训练集、相同的类别加权与相同的测试轨迹上评估：

### 6.1 外部 SOTA 与系统基线

| 类别 | 方法名 | 刊物层级 | 忠实复现细节与评测公平性说明 |
|---|---|---|---|
| **写寿命 / 分流** | **`PCStream`** | FAST '19 (正刊) | 使用真实内核采集的用户态调用栈哈希 `(app_idx, u_stack_id)`（体现跨二进制虚拟地址空间不碰撞）。① **Zero-Shot**：未见新应用 PC 0 命中，回退训练全局多数类；② **Online Death**：严格在文件真实死亡时刻 `t_death` 到达后才更新该 PC 的平均寿命估计。 |
| **写寿命 / FDP** | **`eFDP` / `eFDP+`** | arXiv (`2611.06461`, 2026-11) | 复现响应式死亡直方图 + 周期性 RUH 重规划：① **`eFDP-Paper (420s)`** 还原原论文约 7 分钟重规划滞后；② **`eFDP-Agile (30s / 60s)`** 强行加速 7–14× 的超强敏捷对照；在 Exp 1 50 分钟仿真中进一步构建 **`eFDP+`**（额外白送强分箱键 `(pc_sig, dir_churn_bucket)` 且放长阶段至 600s 让其跨过 420s 重规划窗口！）。 |
| **内核无状态 ML** | **`Heimdall` (Faithful)** | EuroSys '25 (正刊) | 严格按原论文无状态块层 3 层标量 MLP（`14 -> 64 -> 64 -> 32 -> out`, `LayerNorm + ReLU + AdamW`），输入当前孤立事件的 14 维标量（`op, is_seq, mode_changed, log(fsz), log(off), off/fsz, log(cnt), stride_sign, log(|stride|), open_flags(4), region`），无跨步隐状态，与 KSSM 共享完全相同的训练集和类均衡损失。 |
| **增强上界对照** | **`Heimdall+`** | 内部强对照 | 在 `Heimdall` 基础上**额外白送 KSSM 独有的两个跨时间汇总状态特征**（尺度不变相对目录周转率 `rel_churn` + 线程历史读写比 `thread_rw_ratio`），用于剥离“特征工程”与“序列隐状态”的各自净贡献。 |
| **块层马尔可夫** | **`BASIC-Prefetcher`** | SYSTOR '26 (正刊) | 基于 `basic_tok = (s1 << 5) \| s4`（原语 + 请求大小 + 步长桶）统计按线程一阶马尔可夫转移条件概率 $P(y \mid \text{tok}_{t-1}, \text{tok}_t)$，带 Laplace 平滑与 1-gram backoff。 |
| **审稿终极消融** | **`Cross-Entity EWMA + 3L-MLP`** | 强消融基线 | **回应顶会审稿人最锋利反问**：赋予其与 KSSM **100% 相同的 5 元相对词表（192 槽）与 `task`/`inode` 跨实体状态绑定**，仅将输入相关 Selective SSM 替换为固定衰减率 EWMA 聚合（测了 `Single α=0.80`、`Multi 4-scale [0.5, 0.75, 0.875, 0.9375]`、`UltraSlow 5-scale [+0.96875]` 三档！）+ 3 层强非线性浮点 MLP（`192 -> 128 -> 64 -> out`，~37K MAC，物理上无法 `<60ns` 进 eBPF 内核）。 |
| **读路径页缓存** | **`Linux Default LRU` / `Heuristic_Size` / `Heuristic_SeqDrop`** | OS 基线 + 强启发式 | ① `Linux 5.15 Default` 原生两链表 LRU；② `Heuristic_Size`（传统静态规则 `size >= 32 MB` 盲目 `drop-behind`）；③ **`Heuristic_SeqDrop`**（强手工顺序读规则：连续 4 个 128 KB 顺序窗 + 16 MB 预读安全垫即 `drop-behind`，能保护随机 RocksDB 但会误杀可重用热顺序读！）。 |
| **SSD 下/上界** | **`No-Stream` / `Oracle`** | FTL 基准 | `No-Stream` 全部写入单 RUH；`Oracle` 拥有上帝视角按精确真实死亡时间分入 3 个独立 FDP RUH，标定物理 WAF 下界。 |

### 6.2 KSSM 内部系统与算法消融矩阵

- **实体与耦合消融**：`KSSM-InodeOnly`（彻底移除 `h_task`，单实体读出）→ `KSSM-Uncoupled`（保留 `h_task` 但强制 $\gamma=\eta=0$ 无门控交叉耦合）→ `KSSM-TrueSelective-Float64D`（完整双向耦合 + 连续动态选择性门控）。
- **核内定点与选择性消融**：`Exact Q7.8 64D (1,003 ns)` → `Shift-Gate 64D (653 ns)` → `Naive SWAR (271 ns)` → `Fixed-Decay Guard-SWAR (45.9–58.1 ns)` → `Selective Var-Shift (87.4 ns, rcx 瓶颈)` → **`4-Band / 8-Band Independent Selective Guard-SWAR (37.8–53.3 ns)`**。
- **训练与慢回路消融**：`No Hard-STE` vs `Hard-STE QAT` vs `MSE Gate Distillation`；`KSSM-ZeroShot (冻结)` vs `KSSM-Full (+SlowLoop 零 BPTT 校准 <1KB 头)`；`Shared-Inode Gating` vs `Common-LRU` vs **`TL-PerCPU Gating`**。

---

## 七、全阶段实测结果与深入分析（Exp 0-A → Exp 4）

### 7.1 Exp 0-A：Linux 5.15 Verifier、纳秒微基准与 4KB 高并发 CPU 开销（`exp0a/`）

#### （1）Verifier 静态复杂度与零浮点 JIT 验证
- 全部 5 类生产/LSM 钩子在未修改 Linux `5.15.0-177-generic` 上一次 `[PASS+ATTACH]`：
  - `lsm_kssm_native_swar_64d` (`TASK_STORAGE + INODE_STORAGE`)：`960` processed insns (`0.10%` of 1M), `1029` xlated / `4588 B` JIT
  - `ext4_kssm_read` / `ext4_kssm_write` (v5)：`1,978` / `2,005` insns (`0.20%`), `5213 B` / `5256 B` JIT
  - `vfs_kssm_open` / `vfs_kssm_unlink` (v5)：`1,388` / `57` insns, `5929 B` / `356 B` JIT
- 对导出的 `ext4_kssm_write` 真实 x86_64 JIT 二进制（`5,256 B`）执行 `objdump -D -b binary -m i386:x86-64` 全量扫描：**总计 `1,451` 条机器指令，FPU / SSE / AVX / AMX 指令 = `0` 条**。

#### （2）核内单次双状态更新纳秒微基准演进表（`BPF_PROG_TEST_RUN`, CPU 2 稳态 `2.10 GHz`, warmup 200K + 7×500K 中位数）

| SSM 核实现变体 | 隐状态 | 查表与算术机制 | Verifier 指令数 | xlated / JIT | **净单次耗时 (ns)** | 红线 `<60ns` |
|---|---:|---|---:|---:|---:|---|
| `Exact Q7.8 64D` (原标量 + 双读出头) | 128 B | 5×2 稠密表 + `4×64` 标量 `imul` + 无分支 `sat16` | `28,961` | `230` / `1192 B` (64轮) | `1,002.89 ns` | ❌ 超标 16.7× |
| `Exact Q7.8 64D` (纯 SSM 无读出) | 128 B | 同上，去读出头 | `15,458` | `165` / `837 B` | `680.64 ns` | ❌ 超标 11.3× |
| `Shift-Gate 64D` (标量移位, 纯 SSM) | 128 B | 标量二项移位去 `imul` | `27,530` | `174` / `828 B` | `652.75 ns` | ❌ 超标 10.9× |
| `Exact Q7.8 32D` (32 维标量对照) | 64 B | 32 维标量 `imul` + 双读出 | `14,433` | `230` / `1189 B` | `499.45 ns` | ❌ 超标 8.3× |
| `Naive SWAR 4-Lane 64D` | 128 B | 4 路打包，5×2 表 + 5 指令 SWAR 加 | `3,313` | `1037` / `4300 B` | `271.09 ns` | ❌ 超标 4.5× |
| `Guard-SWAR 4-Lane 64D (Dual 48KB)` | 128 B | Excess-16384 收缩不变集，独立双表 | `966` | `1039` / `4055 B` | `80.24 ns` | 🟡 略超 1.3× |
| `Guard-SWAR 4-Lane 64D (Shared 24KB)` | **128 B** | 完整 5 槽稠密表常驻 L1d，固定移位 | `695` | `779` / `3290 B` | **`58.14 ns`** | ✅ **PASS** |
| `Guard-SWAR 4-Lane 64D (Subspace)` | **128 B** | 4×16D 子空间因子化，固定移位 | `546` | `613` / `2535 B` | **`45.94 ns`** | ✅ **PASS** |
| `Guard-SWAR 8-Lane 64D (Q0.6)` | 64 B | 8 路 `int8` 打包对照 | `377` | `421` / `1788 B` | **`32.43 ns`** | ✅ **PASS** |
| `Selective Var-Shift 64D` (反面坑) | 128 B | 运行时可变移位触发 x86 `rcx` 瓶颈 | `727` | `733` / `3594 B` | `87.35 ns` | ❌ 超标 |
| `4-Band Selective Guard-SWAR` (混合 / 全激活) | **128 B** | 4 子带独立 `HOLD/SLOW/FAST/RESET` 编译期移位 | `1,493` | `1623` / `6525 B` | **`37.83 / 50.19 ns`** | ✅ **PASS** |
| `8-Band Selective Guard-SWAR` (混合 / 全激活) | **128 B** | 8 子带独立 4 档编译期移位 | `1,444` | `1533` / `6197 B` | **`38.02 / 50.27 ns`** | ✅ **PASS** |
| **`Unified Pruned g_s1[32][4] 4-Band` (10-App 真权重)** | **128 B** | **生产最终核（`24.7 KB` 底座, 128 B 门控）** | `5,328` | `1753` / `7062 B` | **`53.29 ns`** | ✅ **PASS** |

- 附加拆解：单任务 `HOLD` 噪声跳过路径仅 **`3.47 ns`**，`PHASE_RESET` 路径仅 **`28.31 ns`**；8 通道单任务线性读出头占 `+28.2 ns`（仅在决策点按需调用）。

#### （3）`/mnt/nvme12` 高并发 4KB `fio` CPU 开销实测（`numjobs=4, bs=4k, psync`, `kernel.bpf_stats_enabled=1`）
- **翻车对照（v4 朴素步长门控）**：在 `randwrite_dio` 下触发率达 `99.74%`，单次 `bpf_get_stackid()` 消耗 `632–778 ns`，导致 `ext4_kssm_write` 均摊 `880–917 ns`，BPF CPU 占比飙升至 **`6.88% – 8.57%`**（与 `eFDP` 8–9% 相当 ❌）。
- **修正后（v5 Mode-Transition + 128 KB Byte-Budget Gating，完整保留 `bpf_get_stackid` 16 槽栈环）**：

| 4KB 负载 (`numjobs=4`) | 配置 | IOPS | 均摊单次耗时 | 触发率 | **数据路径 BPF CPU 占比** | 吞吐比 |
|---|---|---:|---:|---:|---:|---:|
| `randwrite_dio` (Direct 随机写, 对标 `eFDP`) | Baseline / KSSM(Subspace) / KSSM(Shared) | `255.9K` / `252.9K` / `241.3K` | — / `92.9 ns` / `95.7 ns` | `3.56%` / `3.65%` | **`0.89%`** / **`0.86%`** | `0.99×` / `0.94×` |
| `randread_dio` (Direct 随机读) | Baseline / KSSM(Subspace) / KSSM(Shared) | `54.8K` / `54.6K` / `54.8K` | — / `155.2 ns` / `150.4 ns` | `3.67%` / `3.61%` | **`0.52%`** / **`0.54%`** | `1.00×` / `1.00×` |
| `seqwrite_buf` (Buffered 顺序写) | Baseline / KSSM(Subspace) / KSSM(Shared) | `702.0K` / `695.7K` / `663.3K` | — / `79.1 ns` / `78.6 ns` | `3.35%` / `3.44%` | **`1.18%`** / **`1.17%`** | `0.99×` / `0.95×` |
| `seqread_buf` (纯内存极限热读) | `Empty fentry` / KSSM(Subspace) / KSSM(Shared) | `1.55M` / `1.54M` / `1.52M` | `21.3 ns` / `84.1 ns` / `84.5 ns` | `3.25%` / `3.26%` | `0.54%` 底噪 → 净增 **`1.67%` / `1.59%`** | `1.00×` / `1.00×` |

---

### 7.2 Exp 0-B：5 应用 LOAO 零样本泛化与负载切换失明期对决（`exp0b/`）

#### （1）5-Fold Leave-One-App-Out 零样本宏平均（未见新应用第 0 秒 `open(O_CREAT)`）

| 方法 | 刊物层级 / 延迟 | Life0s Acc | **Life0s BalAcc** | **Life0s MacroF1** | **One-Shot Scan AUC** |
|---|---|---:|---:|---:|---:|
| `PCStream` | FAST '19 (正刊) | `70.52%`* | `33.33%` | `27.17%` | N/A |
| `Heimdall` (Faithful 无状态块层 MLP) | EuroSys '25 (正刊) | `29.43%` | `33.33%` | `13.77%` | `0.9016` |
| `Heimdall+` (超强对照：白送相对周转率+读写比) | 内部强对照 | `38.71%` | `47.61%` | `29.00%` | `0.9510` |
| `BASIC-Prefetcher` | SYSTOR '26 (正刊) | `41.39%` | `40.42%` | `31.61%` | `0.8271` |
| `KSSM-InodeOnly` / `KSSM-Uncoupled` / `KSSM-Float64D` | 非选择性浮点消融 | `53.2% / 51.9% / 51.4%` | `50.3% / 48.4% / 46.7%` | `36.6% / 35.6% / 34.7%` | `0.907 / 0.888 / 0.895` |
| **`KSSM-GuardSWAR-Shared`** (bit-exact `58.1 ns`) | **核内真原型** | **`52.29%`** | `48.14%` | **`38.20%`** | `0.9174` |
| **`KSSM-GuardSWAR-Subspace`** (bit-exact `45.9 ns`) | **核内真原型** | `51.33%` | **`51.72%`** (`+18.4%`) | `37.25%` | **`0.9329`** (`>0.92` ✅) |
| **`True Selective KSSM (Float64D)`** (Exp 0-B+) | **输入相关动态门控** | **`58.72%`** | **`58.05%`** (`+24.7%`) | **`44.61%`** | `0.8360` |

*\*注：`PCStream` 的 `70.52%` Acc 源于跨二进制 0 命中全部回退至多数类 Short，其 3 类真实分辨力 `BalAcc = 33.33%`。*

#### （2）负载切换冷启动失明期对决（未见新应用启动后各时间窗 Balanced Accuracy）

| 方法 | 刊物层级 | `0–10s` (切换瞬间) | `10–30s` | `30–65s` | `Full 0–65s` |
|---|---|---:|---:|---:|---:|
| `eFDP` (论文 420s 重规划滞后) | arXiv (`2611.06461`) | `33.33%` | `36.67%` | `33.33%` | `33.33%` |
| `eFDP` (激进 30s 重规划) | arXiv (`2611.06461`) | `33.33%` | `36.67%` | `33.33%` | `33.33%` |
| `PCStream` (在线死亡适应) | FAST '19 (正刊) | `33.33%` | `36.67%` | `33.33%` | `33.33%` |
| **`KSSM-GuardSWAR` (纯零样本冻结)** | **Ours (Zero-Shot)** | **`46.91%`** (`v1: 51.3%`) | **`45.24%`** | `39.42%` (`v1: 47.2%`) | **`48.14%`** (`v1: 56.2%`) |
| **`KSSM-GuardSWAR (+SlowLoop 零 BPTT)`** | **Ours (冻结底座+<1KB头)** | **`55.25%`** | **`55.03%`** | **`66.97%`** (`v1: 78.3%`) | **`64.48%`** |

---

### 7.3 Exp 0-B+ & Exp 2-Part1：审稿人终极消融（`Cross-Entity EWMA + 3L-MLP` vs `True Selective KSSM` + Hard-STE QAT）

**战场 2（`exp0b_selective/` & `exp2_scale_distill/`）**：真实内核采集的 `mp_train` (`383,655` 事件, `K∈{0,3,8}`) → `mp_test` (`555,608` 事件, `K∈{0,5,15,35}`, 0 丢包)。3 类目标文件严格**同目录、同大小（512 KB）、同打开标志、同一创建线程**，仅靠穿插 $K$ 条真实日志追加 / 心跳配置读 / 指标覆写噪声之前的上游阶段区分：

| 模型（仅在 `K∈{0,3,8}` 上训练，全流式外推测试 `K=0, 5, 15, 35`） | `K=0` | `K=5` | `K=15` | **`K=35` (OOD 重噪声)** | **总体 BalAcc** | HOLD 跳过率 |
|---|---:|---:|---:|---:|---:|---:|
| `Heimdall` (Stateless 3L-MLP, `L=1`) | `71.75%` | `51.91%` | `52.33%` | `50.93%` | `56.65%` | — |
| `Cross-Entity Single-EWMA (α=0.80) + 3L-MLP` | `75.32%` | `71.67%` | `73.13%` | **`34.03%`** (崩盘) | `63.67%` | `0%` |
| `Cross-Entity Multi-EWMA (4-scale) + 3L-MLP` | `75.41%` | `46.30%` | `74.90%` | **`34.52%`** (崩盘) | `57.90%` | `0%` |
| `Cross-Entity UltraSlow-EWMA (5-scale, α=0.96875) + 3L-MLP` | `75.60%` | `50.88%` | `72.90%` | **`34.59%`** (崩盘) | `58.61%` | `0%` |
| `Fixed-Decay Guard-SWAR 64D` (非选择性固定移位) | `70.20%` | `36.69%` | `64.68%` | **`34.05%`** (崩盘) | `51.45%` | `0%` |
| `1-Global-Gate Selective SWAR` (粗糙单标量 STE) | `73.69%` | `72.45%` | `54.53%` | `32.96%` (量化失败) | `58.42%` | `8.6%` |
| `4-Band Selective SWAR` (仅软退火无 Hard-STE) | `69.85%` | `65.48%` | `67.58%` | `56.29%` | `64.81%` | `32.9%` |
| `4-Band Selective SWAR` (+ Naive MSE 门控蒸馏) | `67.85%` | `57.19%` | `65.64%` | `36.95%` (破坏 HOLD!) | `57.02%` | `22.8%` |
| `8-Band Selective SWAR + Hard-STE QAT` (`38.0–50.3 ns`) | `70.11%` | `58.19%` | `72.75%` | `62.79%` | `65.99%` | `37.9%` |
| `True Selective KSSM (Float64D)` (连续动态门控) | **`75.32%`** | `70.51%` | **`75.58%`** | **`62.58%`** (`+28.0%`) | `71.03%` | — |
| **`4-Band Selective SWAR + Hard-STE QAT` (`37.8–50.2 ns`)** | `74.22%` | **`70.61%`** | `73.75%` | **`70.80%`** (**`+36.3%` vs EWMA / `+8.2%` 反超浮点!**) | **`72.35%`** | **`32.5%`** |

- **机理结论**：固定衰减 EWMA 陷入无法调和的 **Stability-Plasticity 两难**——快尺度被 35 条噪声冲刷归零（$0.75^{35} \approx 4\times 10^{-5}$），慢尺度把前后阶段糊成浆糊且仍被噪声淹没 90% 能量。而 Hard-STE QAT `4-Band Selective SWAR` 自动将 `32.5%` 噪声路由至硬件零泄漏 `HOLD (α=1.0)`，连连续 `sigmoid` 残余衰减泄漏（$0.98^{35}\approx 0.49$）都一并消除！

---

### 7.4 Exp 0-C & Exp 4：读路径端到端混部系统收益与锁频三元对抗压测（`exp0c/`, `exp4_read_hardened/`）

#### （1）Exp 0-C：双场景二元混部（`fresh 8 GiB cgroup v2`, 每组冷起跑一字节不差 `file = 5.11 GiB, pgscan = 0`）

| 实验组 (`8 GiB cgroup v2`, 35s) | FG `RocksDB` QPS | P50 (µs) | P95 (µs) | **P99 (µs)** | **P99.9 (µs)** | **BG LLM (MB/s)** | BG Ckpt (MB/s) | cgroup `pgscan` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `solo` (无后台独享上限) | `974,825` | `4` | `5` | `6` | `12` | — | — | `0` |
| `base_llm_only` (Linux 默认 LRU + 纯 LLM) | `235,401` | `6` | `86` | **`127`** | `303` | `979.6` | — | `9,472,634` |
| **`kssm_llm_only`** (100% 纯神经读出闭环) | **`739,590`** (`3.14×`) | `5` | `8` | **`10`** (**`-92.1%`**) | **`20`** (**`-93.4%`**) | **`1,608.6`** (**`+64.2%`**) | — | **`9,620`** (`-99.9%`) |
| `base_mixed` (Linux 默认 LRU + LLM + Ckpt) | `235,483` | `6` | `88` | **`163`** | `342` | `1,012.6` | `175.5` | `11,224,694` |
| `heur_mixed` (静态规则 `size>=32MB` 盲目 drop) | `82,612` (`-65%`) | `15` | `109` | **`198`** (`+21.5%`) | `331` | `1,399.6` | `175.5` | `0` (`SST FP=82`) |
| **`kssm_mixed`** (KSSM 实时闭环) | **`1,088,926`** (`4.62×`)* | `3` | `5` | **`7`** (**`-95.7%`**) | **`13`** (**`-96.2%`**) | **`1,914.2`** (**`+89.0%`**) | **`175.5`** | **`0`** (`SST FP=0`) |

*\*注：`kssm_mixed` 在 `ondemand` 调频下 QPS 略超 `solo` 已查明为热态升频效应，并在下述 Exp 4 中锁死 `performance` governor 彻底消除。*

#### （2）Exp 4：锁频 `performance`（CPU `0..15`）三元混部硬刚强手工规则 `Heuristic_SeqDrop`

场景：**FG-1 `RocksDB 5.2 GiB` 随机点查 + FG-2 `HotSeqEmbed 512 MiB` 高频热顺序扫描（455 遍 = `227.5 GiB`）+ BG `43 GB .safetensors` 一次性流式加载**，严苛锁频 + `fresh 8 GiB cgroup v2`：

| 对比组 (`CPU 0..15 performance`, `8 GiB cgroup`) | **FG-1 `RocksDB` QPS** | **FG-1 P99 (µs)** | **FG-2 热顺序扫 (MB/s)** | **FG-2 P99 (ms)** | **BG LLM (MB/s)** | 逐文件误杀审计 (`Emb` / `RocksDB .sst`) |
|---|---:|---:|---:|---:|---:|---|
| `solo_fg` (双前台独享上限) | `1,262,311` | `4` | `6,690.9` | `73.5` | — | — (`pgscan=0`) |
| `base_lru` (Linux 5.15 默认 LRU) | `321,287` | **`116`** | `6,522.1` | `110.1` | `1,794.5` | — (`pgscan = 17,456,128` 页!) |
| `heur_size` (静态 `size >= 32 MB`) | `83,173` | `179` | `2,281.2` | `213.9` | `2,450.7` | `Emb=1(157.15G)`, `SST=82(20.15G)` |
| **`heur_seqdrop`** (**强手工顺序规则**) | `1,164,186` | `7` | **`2,412.3`** (`-64%`!) | **`201.5`** (`2.7×`!) | `2,524.8` | **`Emb=1(79.19G` 误杀!)**, `SST=0(0.00G)` |
| **`kssm_adaptive`** (**KSSM 自适应闭环**, `98.1 ns`) | **`1,124,105`** (`89.1%` solo) | **`8`** (**`-93.1%`**) | **`6,652.4`** (**`2.76×`** vs SeqDrop!) | **`75.1`** (**`-62.7%`** vs SeqDrop!) | **`2,753.8`** (**`+53.5%`** vs LRU / **`+9.1%`** vs SeqDrop) | **`Emb=1(0.48G = 0.21%)`**, **`SST=0(0.00G)`** |

- **三方全赢底层机理**：`KSSM Adaptive` 通过 1:1 全局 `8 GiB` 2 MiB 簇重用指纹表在第 1 遍试探（仅 `0.48 GiB`）后即自监督认出 `FG-2` 是可重用热顺序读，随后 **454 遍零驱逐**，使 `5.2 + 0.5 = 5.7 GiB` 双前台热工作集 100% 锁留 DRAM（`pgscan=0`），既打穿了 `Heuristic_SeqDrop` 的 `2.76×` 误杀短板，又因 `FG-1/FG-2` 均不掉盘抢 NVMe 带宽而把后台大模型加载推至全场最高 **`2,753.8 MB/s`**！

---

### 7.5 Exp 1 & Exp 2-FDP 回灌：写路径真机闭环与 50 分钟动态切换 NVMe FDP 仿真（`exp1_write_fdp/`）

#### （1）真机 Linux 5.15 写路径 `F_SET_RW_HINT` + 零 BPTT 慢回路实测闭环
- `kssm_write_live.bpf.c` + `kssm_write_live_daemon.c`（`24 KB` 冻结底座 + `784 B` 线性寿命头）：一次过 5.15 Verifier；`/mnt/nvme12/yjw_kssm_exp1` 实测 `births = 40` → **`F_SET_RW_HINT applied = 40` → `F_GET_RW_HINT verified = 40`（100% 写入内核 `inode->i_write_hint` 验证通过）**；`deaths_feedback = 40` → 每 2 秒自动刷回 `784 B` 线性头至内核 `head_map`（`syncs = 7`）；`on_vfs_open` 均摊仅 **`87.78 ns`**。

#### （2）50 分钟（`3,000s`, `149.82 GiB`）5 应用动态切换 NVMe FDP (4-RUH + GC_RUH) 仿真
严格 5-Fold LOAO 零泄露切入（每切换新应用强行重置为未见过该应用的 LOAO 权重）；95% 峰值满载压力（`Logical = 6.0 GiB`, `Static = 0.8 GiB`, `25×` 满盘覆写）；为 `eFDP+` 额外赋予强分箱键 `(pc_sig, dir_churn_bucket)`：

| OP 档位 | 分流策略 | 50-Min BalAcc | 全周期 WAF | 切换窗 `0–420s` WAF | **GC 搬迁 (GiB)** | 擦除数 | **P99 写延迟 (µs)** | P99.9 (µs) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **`OP = 7%`**<br>(高密/消费盘) | `No-Stream (Single-RUH)` | `0.0%` | `1.386` | `1.357` | `57.87 G` | `3,240` | `14,811` | `39,906` |
| | `Heimdall` (EuroSys '25) | `26.3%` | `1.370` | `1.343` | `55.37 G` | `3,200` | `13,411` | `40,046` |
| | `PCStream` (FAST '19, Online) | `42.5%` | `1.409` | `1.365` | `61.30 G` | `3,296` | `15,091` | `55,201` |
| | `eFDP+` (`arXiv:2611.06461`, `420s`) | `38.4%` (`0%` 盲窗) | `1.259` | `1.249` | `38.85 G` | `2,937` | `7,076` | `40,466` |
| | `eFDP+` (7× 敏捷加速版, `60s`) | `62.1%` (`0%` 盲窗) | `1.243` | `1.225` | `36.43 G` | `2,897` | `6,516` | `39,346` |
| | `KSSM-ZeroShot` (旧 4-App 训练) | `48.5%` | `1.385` | `1.334` | `57.66 G` | `3,237` | `14,671` | `40,316` |
| | **`KSSM-ZeroShot` (新 9-App `4-Band`)** | **`69.2%`** | `1.264` | **`1.241`** (**反超 eFDP+**) | **`39.59 G`** (`-31.3%`) | `2,949` | **`6,646`** (**反超 eFDP+**) | `43,406` |
| | **`KSSM-Full (+SlowLoop, 9-App)`** | **`90.2%`** | **`1.138`** | **`1.103`** | **`20.72 G`** (**`-46.7%`** vs eFDP+) | **`2,646`** | **`2,491`** (**`-64.8%`** vs eFDP+) | **`30,631`** |
| | `Oracle` (理论下界) | `100%` | `1.055` | `1.022` | `8.22 G` | `2,447` | `916` | `17,331` |
| **`OP = 14%`**<br>(标准企业盘) | `No-Stream` | — | `1.250` | `1.252` | `37.39 G` | `2,907` | `10,751` | `23,071` |
| | `eFDP+` (`420s`) / `eFDP+` (`60s`) | — | `1.163` / `1.154` | `1.170` / `1.154` | `24.36 G` / `23.02 G` | `2,699` / `2,677` | `5,386` / `4,976` | `22,371` / `22,791` |
| | **`KSSM-ZeroShot` (新 9-App)** | **`69.2%`** | `1.171` | **`1.166`** (**反超 eFDP+**) | `25.55 G` | `2,718` | **`5,386`** (持平在线 eFDP+) | `23,491` |
| | **`KSSM-Full (+SlowLoop)`** | **`90.2%`** | **`1.060–1.062`** | **`1.058–1.059`** | **`9.00–9.36 G`** (**`-63.1%`** vs eFDP+) | **`2,453`** | **`932`** (**`-82.7%`, 追平 Oracle!**) | **`15,511`** |
| | `Oracle` | — | `1.016` | `1.005` | `2.47 G` | `2,349` | `916` | `6,096` |
| **`OP = 28%`**<br>(写密集企业盘) | `No-Stream` / `eFDP+ (420s)` | — | `1.134` / `1.078` | `1.139` / `1.080` | `20.05 G` / `11.74 G` | `2,615` / `2,483` | `5,396` / `3,296` | `15,231` / `14,951` |
| | **`KSSM-Full (+SlowLoop)`** | **`90.2%`** | **`1.008–1.009`** | **`1.008–1.010`** | **`1.26–1.41 G`** (**`-89.3%`** vs eFDP+) | **`2,316`** | **`916`** (**追平 Oracle**) | **`3,576`** |

---

### 7.6 Exp 2-Part2 & Exp 3：10 类真实异构应用 10-Fold LOAO、统一引擎与 1–64 核并发扩展性（`exp2_scale_distill/`, `kssm_engine/`）

#### （1）10-App 10-Fold LOAO 跨二进制零样本宏平均 & 原 5 应用 `4→9` Scaling Law

| 方法 | **10-App Acc** | **10-App BalAcc** | **10-App MacroF1** | **10-App Scan AUC** | 原 5 应用 BalAcc (`4→9` 扩容效应) |
|---|---:|---:|---:|---:|---:|
| `PCStream` (FAST '19) | `69.47%`* | `33.33%` | `27.01%` | N/A | `33.33% → 33.33%` |
| `Heimdall` (EuroSys '25, Faithful) | `34.58%` | `37.80%` | `19.05%` | `0.8928` | `33.33% → 33.33%` |
| `BASIC-Prefetch` (SYSTOR '26) | `45.89%` | `42.93%` | `34.07%` | `0.8724` | `40.42% → 37.60%` |
| `KSSM-InodeOnly` (无 `h_task` 消融) | `70.17%` | `63.11%` | `54.33%` | `0.9079` | `50.30% → 74.59%` |
| `Cross-Entity Multi-EWMA + 3L-MLP` | `70.17%` | `64.11%` | `55.23%` | `0.9219` | `53.01% → 71.45%` |
| `KSSM-FixedSWAR` (`46–58 ns`, Bit-Exact) | `69.85%` | `65.19%` | `56.57%` | `0.9050` | `48.14% → 68.29%` (**`+20.2%`**) |
| **`KSSM-4Band-SelSWAR (g_s1[32][4], 53 ns)`** | **`70.50%`** | **`67.74%`** | **`58.46%`** | `0.9045` | `47.02% → 70.09%` (**`+23.1%`**) |
| `KSSM-Uncoupled` ($\gamma=\eta=0$ 消融) | `72.30%` | `68.28%` | `58.38%` | `0.9019` | `48.39% → 73.93%` |
| **`KSSM-TrueSelective-Float64D`** | **`73.06%`** | **`69.02%`** | **`59.56%`** | `0.9022` | `58.05% → 73.77%` (**`+15.7%`**) |
| **`KSSM-4Band-SelSWAR (+SlowLoop)`** | **`87.30%`** | **`73.76%`** | **`72.54%`** | `0.8932` | `64.48% → 73.24%` |

- **端到端自然涌现的操作系统门控语义**：10-App 联合训练的 `128 B` `g_s1[32][4]` 门控表在无人工干预下自动学会：`OPEN_CREAT (s1=1) → [3, 3, 1, 0]`、`RENAME (s1=15) → [3, 2, 1, 3]`（`3 = PHASE_RESET` 自动清空旧态高增益继承 `h_task`），`OPEN_RW (s1=6)` / `UNLINK (s1=14)` 自动启用 `0 = HOLD` 保护记忆！

#### （2）统一引擎（`kssm_engine/`）`1 → 64` 核并发扩展性压测（双路 Xeon Gold 6530）

| 压测场景 (`fio bs=4k, psync`) | 核数 | Baseline | **`KSSM TL-PerCPU`** (`BPF_CPU / avg_ns`) | `KSSM SharedInode` | `KSSM CommonLRU` |
|---|---:|---:|---|---|---|
| **A. 独立文件 PageCache 读** | `1` / `16` / `32` / **`64`** | `0.42` / `2.91` / `3.00` / **`3.02 MIOPS`** | `0.37` / `2.90` / `3.00` / **`3.02 MIOPS` (`100.1%`, `2.72%` / `145 ns`)** | — | `3.02 MIOPS` (`3.05%` / `166 ns`) |
| **B. 极端单共享热文件竞争** (64 核死磕 1 文件!) | `4` / `16` / `32` / **`64`** | `0.18` / `0.87` / `2.46` / **`3.45 MIOPS`** | `0.19` / `0.88` / `1.94` / **`3.03 MIOPS` (`+28.9%` vs Shared, `5.04%` / `942 ns`)** | `2.35 MIOPS` (`-31.9%`, `8.18%` / `1,997 ns`) | `2.45 MIOPS` (`-29.0%`, `8.57%` / `2,031 ns`) |
| **C. PM174X NVMe Direct 随机写** | `1` / `4` / `16` / **`32`** | `0.09` / `0.28` / `0.54` / **`0.49 MIOPS`** | `0.09` / `0.28` / `0.53` / **`0.50 MIOPS` (`102%`, `0.68%–2.32%` / `160–195 ns`)** | — | — |

---

## 八、真实科研排坑实录（12 个关键系统/算法陷阱与根治方案）

以下是我们在整个推进过程中亲历、实测捕获并逐一攻克的 **12 个高含金量真问题**（每一条都是顶会论文 *Design Lessons* 或 *Implementation* 章节的极佳素材）：

1. **BPF `512 B` 栈帧溢出**：64 通道全展开时 14 个表基址指针超出 eBPF 10 寄存器引发大量栈 spill，外加 `struct inode_state zero = {}` 占栈 184 B；改用 `zero_ino_map` 只读模板 + SWAR 4 路打包压缩指针压力，栈深压至 `< 104 B`。
2. **Linux 5.15 Verifier `1,000,000` 指令路径爆炸**：条件 `sat16` 与 16 槽栈环 `if` 匹配与后继循环形成 $2^{16}$ 笛卡尔积爆炸（`-7 E2BIG`）；通过全无分支位运算掩码 `sat16`、无分支栈环扫描及 `volatile` per-CPU scratch 截断值域传播，验证复杂度压低 500× 至 `< 2,010 insns`（单核 `546 – 1,493 insns`）。
3. **Linux 5.15 `BPF LSM` 静默空转陷阱**：`CONFIG_BPF_LSM=y` 下 `SEC("lsm/file_open")` 一次过 Verifier 且 `ATTACH OK`，但未设启动参数 `lsm=...,bpf` 时运行时 **`lsm_hits = 0`** 静默不执行；且 5.15 `fentry` 禁调 `bpf_inode_storage_get`。改用 `fentry/ext4_file_{read,write}_iter` + `Per-CPU LRU_HASH` 实现零重启落地。
4. **`2.10 GHz` 标量屋顶线（`1,003 ns`）**：原标量公式虽达 `5.46 IPC`，但物理极限决定了不可能 `< 60 ns`；发明 **Banach 收缩不变集 + Excess-16384 Guard-Bit SWAR** 提速 17–22× 至 `45.9 – 58.1 ns`。
5. **x86_64 `rcx` 可变移位串行瓶颈（`87.4 ns`）**：运行时 `>> g->d_i` 争抢单寄存器 `cl` 使延迟翻倍；改用 **4-Regime 编译期立即数分派**，压低至 **`37.8 – 50.2 ns`**（噪声 `HOLD` 仅 `3.47 ns`）。
6. **`bpf_get_stackid()` 的 `632–778 ns` 隐藏税**：朴素步长门控在随机 4KB 写下触发率 `99.7%`，CPU 飙至 `6.9–8.6%`；升级 **Mode-Transition + 128 KB Byte-Budget Gating** 封顶稳态触发率 `~3.5%`，触盘 CPU 降至 `0.52%–1.18%`。
7. **Ext4 `inode` 号高速复用 + 右截断偏差**：RocksDB 15,175 个创建文件因 Ext4 立刻复用已删 `i_ino` 被合并成 217 个；改用 `(ino, generation)` 唯一实例追踪并严格剔除 22s 尾窗右截断样本。
8. **绝对目录周转率 100× 跨应用量纲漂移 & 离散表 OOV**：改用相对进程因果滑动均值的尺度不变 $s_5$ + 离线 1D Total-Variation 平滑正则（0 运行时开销）。
9. **多线程共读同一 SST 的步长干扰 + `SQLite` 小扫描库标签倒错**：将步长与持续顺序计数移入 `TASK_STORAGE` 按线程追踪；将 `SQLite` 归档扫描表修正为 16×16 MB 一次性全表扫，使 10/10 应用物理标签完全自洽。
10. **连续门控蒸馏反劣化 & Hard-STE QAT 零泄漏反超**：发现 Naive MSE 门控蒸馏破坏硬 `HOLD=0` 导致 `K=35` 从 `70.8%` 跌至 `37.0%`；改用末段 Hard-STE QAT 彻底消除离散休克并物理反超连续浮点 `+8.22%`。
11. **`g_pair[1024]` OOV 误路由 `HOLD`**：精简为 `128 B` `g_s1[32][4]` 后，`log_segment` Acc `+46.5%`、`rocksdb` Acc `+33.7%`。
12. **128 核 `BPF_F_NO_COMMON_LRU` 容量劈分 + 跨文件随机点查击穿快路径 + `ondemand` 频漂**：128 核把 `32,768` 槽 Per-CPU LRU 劈成仅 `256` 槽/CPU（`32 MiB` 窗口导致 `512 MiB` 热顺序文件 0 命中）；改用全局 `4,096 × 2 MiB = 8 GiB` 低频采样表 + 锁定 `performance` governor + 明确 `TL-PerCPU` vs 按 `inode` 门控适用边界后，Exp 4 实现三方全赢（`98.1 ns`, `FG-2` `2.76×` vs `SeqDrop`）。

---

## 九、必须正视的现存局限与审稿风险预判

秉持客观严谨的科研铁律，以下 5 项现存短板必须在论文 *Limitations / Discussion* 中坦诚说明或在投稿前补齐：

1. **真实物理 NVMe FDP / Multi-Stream 独占盘硬件 WAF 实测尚未执行**：受限于 `storage-192` 的 `/dev/nvme7n1` 是多用户共享盘（已用 2.5 TB），Exp 1 在真机上验证了 `F_SET_RW_HINT`（40/40）与零 BPTT 慢回路闭环，而 50 分钟 WAF 与 P99 写尾延迟来自高精度 C++ FDP FTL 仿真器。正式投稿前，若能获得一块可独占整盘 2× Preconditioning 预处理的 NVMe Streams（PM174X）或 NVMe 2.0 FDP 硬件盘（或运行 `NVMeVirt` 内核级 FDP 仿真设备），将使写路径实证从 95 分拉满至 100 分。
2. **Linux `6.4+` 原生 `BPF_MAP_TYPE_INODE_STORAGE` 实机运行态对比缺失**：当前 `storage-192` 运行 Linux `5.15.0-177-generic` 且未开 `lsm=bpf`，因此我们验证了 LSM 原生 `INODE_STORAGE` 过 5.15 Verifier，但运行态采用 `Per-CPU LRU_HASH`；在 `6.4+` 内核上实测原生 `fentry + INODE_STORAGE` 在 64 核单共享文件场景下的表现，将是极佳的补充评估。
3. **64 核极端单共享热文件仍留有 `-12.2%` 慢路径缓存行争用**：在 Exp 3 场景 B 中，64 核跨双路 NUMA 死磕同一文件时，`TL-PerCPU` 将吞吐从 `SharedInode` 的 `2.35 MIOPS` 拉回 `3.03 MIOPS`（vs Baseline `3.45 MIOPS`），剩余 `4.2%` 慢路径对同一 `inode_st` 跨 NUMA 写仍带来 `-12.2%` 损耗（正常多文件并发为 `100%` 零损失）。
4. **读路径 `One-Shot Scan` 纯线性头 vs 非线性 MLP 上界轻微差距**：在 10-App LOAO 上，纯线性读出头的 `KSSM` Scan AUC 为 `0.905`（原 5-App `0.933`），略低于无法进内核的 3 层浮点 `Cross-Entity Multi-EWMA + 3L-MLP` (`0.922`) 约 `0.017`；后续可探索核内零分支二阶逐通道交互（如 `(h_i ⊙ h_t)` 哈达玛积线性读出）进一步补齐。
5. **端到端联合评估中的统一写/读双控同时在线压测**：目前 Exp 0-C/4 验证了读路径三元混部闭环、Exp 1 验证了写路径真机 `F_SET_RW_HINT` + 慢回路与 FDP 仿真、Exp 3 将两者合并为单一 `kssm_engine/` 代码库；正式论文评估中可将单一 `kssm_daemon` 在同时开启读冷降级 + 写流分流 + 慢回路校准的满血状态下跑一遍综合端到端压测。

---

## 十、下一步研究与顶会投稿路线图

全部核心技术可行性验证、算法短板攻坚、统一生产引擎与强基线对抗压测（`Exp 0-A → Exp 4`）**现已全部收官通关**。下一阶段正式迈入 **FAST / OSDI / SOSP 论文成型与冲刺投稿阶段**：

1. **阶段 A（论文级图表与定量分析包，1–2 天）**：
   - 基于 `/home/yjw/workspace/research/KSSM` 已落盘的全部真实实验数据，绘制全套出版级矢量图（PDF/SVG）：
     - Fig 1：OS-ML「延迟 vs. 序列选择性记忆」断层定位图（`Heimdall/LinnOS` vs `Kleio/LAKE` vs `eFDP` vs **`KSSM`**）；
     - Fig 2：Exp 0-A 纳秒演进屋顶线柱状图（`1,003 ns → 37.8 ns`）+ 4KB I/O CPU 开销对比（vs `eFDP 8–9%`）；
     - Fig 3：Exp 0-B+/Exp 2 多阶段重噪声曲线（`K = 0, 5, 15, 35`，展示 `EWMA + 3L-MLP` 崩盘 vs Hard-STE `4-Band Selective SWAR` 反超浮点）；
     - Fig 4：Exp 2 10-App 10-Fold LOAO 分应用雷达/分组柱状图 + `4→9` Scaling Law 跃迁图 + 双实体耦合消融阶梯图；
     - Fig 5：Exp 0-C & Exp 4 锁频三元混部端到端对比图（`RocksDB P99` + `HotSeqEmbed` vs `Heuristic_SeqDrop` + `BG LLM` 吞吐）；
     - Fig 6：Exp 1 50 分钟 5 应用动态切换时间序列 WAF 曲线 + `OP=7%/14%/28%` GC 搬迁与 P99 写尾延迟柱状图；
     - Fig 7：Exp 3 `1 → 64` 核并发扩展性曲线（独立文件 + 64 核单共享文件 `TL-PerCPU` vs `SharedInode` + NVMe 写）。
2. **阶段 B（论文大纲与正文初稿撰写）**：
   - 按 FAST / OSDI / SOSP 顶会叙事结构撰写完整 LaTeX 初稿（Introduction → Motivation & Background → KSSM Design → Contractive Selective Guard-SWAR & eBPF Implementation → Evaluation → Discussion & Limitations → Related Work）。
3. **阶段 C（冲刺加分项，视硬件排期择机补充）**：
   - 若组内有可独占擦写的空 NVMe Streams / FDP 盘（或挂载 `NVMeVirt` FDP 模拟块设备），补充真机独占盘 2× Preconditioned 硬件 WAF 实测；
   - 若有 Linux `6.8+` 测试节点，补充 `fentry + 原生 BPF_MAP_TYPE_INODE_STORAGE` 64 核扩展性对照数据。

---

## 十一、代码目录与一键可复现资产全索引

所有源码、编译产物、真机内核轨迹与导出权重均完整保存在 `storage-192` 的 `/home/yjw/workspace/research/KSSM/`（未改动原有 `/home/yjw/workspace/research/SSD_offload` 或任何共享裸盘）：

```text
/home/yjw/workspace/research/KSSM/
├── KSSM_RESEARCH_REPORT.md              # 本完整研究档案与全阶段实验报告
├── vmlinux.h                            # storage-192 Linux 5.15.0-177 BTF 内核头文件 (2.7 MB)
│
├── exp0a/                               # [Exp 0-A] 5.15 能力探针、Guard-SWAR 核演进与 fio 4KB CPU 开销
│   ├── probe_caps.bpf.c, probe_runtime.bpf.c       # 实测 INODE_STORAGE / lsm_hits=0 / stackid
│   ├── kssm_common.h, kssm_exp0a{,_v2,_v3,_v4,_v5}.bpf.c
│   ├── kssm_exp0a{,_v2,_v3,_v4,_v5}_runner.c       # Verifier/JIT 检查与 BPF_PROG_TEST_RUN 纳秒微基准
│   └── run_fio_bench.sh, run_fio_v5.sh             # /mnt/nvme12 4KB 高并发 CPU 开销与 0 浮点 JIT 反汇编
│
├── exp0b/                               # [Exp 0-B] 7 钩子零丢包采集器与 5 应用 LOAO 零样本评测
│   ├── trace_event.h, kssm_collector.bpf.c, kssm_collector.c
│   ├── workloads/{rocksdb_wl.cpp,sqlite_wl.py,pytorch_ckpt_wl.py,llm_safetensors_wl.py,etl_seqscan_wl.py}
│   ├── traces/{rocksdb,sqlite,pytorch,llm,etl}.{bin,pid,collector.log}
│   ├── prepare_dataset{,_v3}.py, dataset_exp0b{,_v3}.pkl
│   ├── run_exp0b_{benchmark,v2,v3}.py, kssm_baked_weights.npz
│
├── exp0b_selective/                     # [Exp 0-B+] 真·选择性 SWAR 核与 Cross-Entity EWMA+3L-MLP 消融
│   ├── kssm_selective.{h,bpf.c}, kssm_sel4.bpf.c, kssm_sel4band.bpf.c & bench_*.c
│   ├── multiphase_noise_wl.py, traces/mp_{train,test}.{bin,pid,log,meta.csv}
│   └── run_selective_vs_ewma.py, eval_sel4band_mp.py
│
├── exp0c/                               # [Exp 0-C] 8 GiB cgroup v2 读路径端到端混部（RocksDB+LLM+Ckpt）
│   ├── fg_rocksdb_bench.cpp             # 5.2 GiB RocksDB 8.9.1 构建与 4 线程 Zipfian 延迟压测
│   ├── bg_llm_ckpt.py                   # 43 GB .safetensors 流式加载 + 周期 1 GiB Checkpoint 写入
│   ├── kssm_exp0c_shared.h, kssm_exp0c.bpf.c, kssm_exp0c_daemon.c, kssm_weights.bin
│   └── run_exp0c.sh, run_exp0c_clean.sh
│
├── exp1_write_fdp/                      # [Exp 1] 写路径真机 F_SET_RW_HINT 闭环 + 50 分钟 NVMe FDP 仿真
│   ├── kssm_write_shared.h, kssm_write_live.bpf.c, kssm_write_live_daemon.c, kssm_life_{base,head}.bin
│   ├── gen_fdp_workload_and_preds.py, shift_{20,50}min_stream.bin
│   └── fdp_ftl_sim.cpp, fdp_ftl_sim     # 高精度 C++ NVMe FDP (4-RUH+GC_RUH) SSD FTL 仿真器
│
├── exp2_scale_distill/                  # [Exp 2] 8-Band/4-Band Hard-STE QAT + 10 类真实异构应用 LOAO
│   ├── kssm_sel8band.bpf.c, bench_sel8band.c, distill_sel8band_mp.py
│   ├── workloads/{compiler_build_wl.py,git_vcs_wl.py,log_segment_wl.cpp,zstd_archive_wl.py,vllm_kv_swap_wl.py}
│   ├── collect_new5.sh, traces/ (10 apps), prepare_10app.py, dataset_10app.pkl (57 MB)
│   └── run_10app_loao.py, loao_10app_results.pkl, shift_50min_9app_sel4.bin
│
├── kssm_engine/                         # [Exp 3] 统一生产级单一 KSSM 系统代码库 + 1-64 核扩展性基准
│   ├── Makefile                         # 一键 make && sudo ./bin/kssm_daemon --verify
│   ├── include/kssm_abi.h               # 统一二进制 ABI (24,704 B base + 1,040 B dual head)
│   ├── bpf/kssm_engine.bpf.c            # 统一 6 钩子 eBPF 引擎 (g_s1[32][4] + TL-PerCPU + 双头 + 慢回路)
│   ├── src/kssm_daemon.c                # 统一用户态控制 + 零 BPTT 慢回路守护进程
│   ├── train_unified_10app.py, models/kssm_{base,head}.bin
│   └── bench/run_scalability.sh         # 1-64 核三场景并发扩展性压测脚本
│
└── exp4_read_hardened/                  # [Exp 4] 锁频三元对抗压测 (vs Heuristic_SeqDrop)
    ├── kssm_exp4_shared.h, kssm_exp4.bpf.c, kssm_exp4_daemon.c
    ├── fg2_hot_seq_embed.py             # 512 MiB 可重用热顺序 Embedding 扫描负载
    └── run_exp4_hardened.sh, run_exp4_kssm_retest.sh
```

### 一键快速复现核心命令参考
- **统一引擎编译、5.15 Verifier 6 钩子检查与 `53 ns` 微基准**：
  ```bash
  cd /home/yjw/workspace/research/KSSM/kssm_engine && make verify
  ```
- **1–64 核并发扩展性压测（独立文件 / 64 核单共享热文件 / PM174X 随机写）**：
  ```bash
  /home/yjw/workspace/research/KSSM/kssm_engine/bench/run_scalability.sh
  ```
- **锁频 `performance` 三元读路径混部（RocksDB + HotSeqEmbed + 43 GB LLM vs `Heuristic_SeqDrop`）**：
  ```bash
  /home/yjw/workspace/research/KSSM/exp4_read_hardened/run_exp4_hardened.sh
  ```
- **50 分钟跨应用动态切换 NVMe FDP 写放大与 P99 写尾延迟仿真（OP=7%/14%/28%）**：
  ```bash
  /home/yjw/workspace/research/KSSM/exp1_write_fdp/fdp_ftl_sim /home/yjw/workspace/research/KSSM/exp2_scale_distill/shift_50min_9app_sel4.bin 96 12
  ```
- **10 类真实应用 10-Fold LOAO 全量零样本评测（严格 GPU 3）**：
  ```bash
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 /home/yjw/miniconda3/envs/sglang/bin/python -u /home/yjw/workspace/research/KSSM/exp2_scale_distill/run_10app_loao.py
  ```
