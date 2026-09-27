# 原源码增量接口：Q20 / B200

## 范围与验证状态

本目录放在原始压缩包解压后的根目录，与 `GPTFuzz-master`、
`pair_official_autodl`、`JailbreakingLLMs-official` 并列。
不覆盖、不删除、不重写原包任何代码或历史结果。

- 已复用：原 `gptfuzz.py:main`、原 LocalLLM、原 PAIR 适配器、原两种离线评估入口。
- 已增加：三种外部方法接入、统一参数入口、目标响应逐条落盘、有限预算、源文件指纹、增量任务列表。
- 默认任务：TAP、ReNeLLM、DeepInception × seeds 100/200/300，共 9 个主条件单元。
- 已有 RA-CoFuzz、Strict GPTFuzzer、PAIR 不在默认增量列表中；仍可手动调用接口。
- Q20/B200、128 tokens、CPU predictor、22GiB 保持固定。本地生成使用原加载器的 Greedy。
- 新 seed 明确在调用原入口前重设 Python/NumPy/Torch；不再把三次同 seed 执行称为独立种子。
- 原算子、在线评分规则、随机模板池和旧评估解析保持原样；不假称与此前 v0.2/v0.3 修订版完全同协议。

本地已做的是**接口/契约单元测试与原文件不变检查**，不是 GPU/API 端到端测试。
当前共 78 项本地测试，包含 7 项 `generate.calls` 统计兼容回归、3 项 TAP 模板选择回归、12 项 ReNeLLM 流程回归
和 30 项 TAP 解析/空候选/单候选兼容回归。
外部依赖没有包含在原压缩包中；必须在服务器运行 `audit --runtime`，
它验证真实构造器和首次生成调用边界，并用中性替身比较已安装 ReNeLLM 的外层控制流程。
TAP 另用中性固定返回值测试真实生成器、解析器与筛选器的七个解析边界、空集恢复和单候选路径。
以上均不调用真实 API，也不是完整搜索的端到端测试。
没有完整单元实测之前，不要直接启动九单元矩阵。

## 与上游方法的区别（需记入论文配置）

TAP、DeepInception 仍通过 EasyJailbreak 的 `single_attack` 执行。
ReNeLLM 恢复服务器原 `attack()` 中逐题的 `single_attack → evaluator → 成功则停止` 循环；
每轮仍传入同一个原 instance，保留其原地变化，不擅自用上一轮返回的 child 替换它。
每个问题最多 10 次本地目标调用，
每个单元最多 200 次；保留方法本身的提前结束，记录实际消耗。
这是统一有限预算下的适配版本，不是上游默认无限制运行。
ReNeLLM 同时遵守原 `evo_max` 上限，最后一次目标调用仍执行内部评估；内部成功标记
只用于原方法停止条件，不替代统一离线指标。新增元数据记录每题轮数、评估数和停止原因。
原外层的汇总日志/结果聚合由本接口的全响应记录和统一离线评估替代。
DeepInception 原外层仅遍历问题、收集结果并作事后评估，未包含按评估反馈重试的循环；
不为了用完 200 次预算添加重复生成。
此前 ReNeLLM 仅单次调用的结果标为旧单轮变体，完整保留但不当作恢复循环后的结果。
原始问题、生成内容、回答只写入私有结果文件，终端仅显示状态/计数/错误位置。
预算在模型调用之前检查，完成的每条响应立即落盘，不能把最后一个树节点当作全部响应。
TAP 模板仅兼容 `reference_responses` → `target_str` 占位符别名，不重写正文。

### TAP 模板固定策略

安全结构诊断显示，seeds100/200 均选择同一条 5990 字符模板且工作流完成；seed300
选择了另一条 248 字符模板，该模板生成的三条探测结果均不包含完整映射或解析器必需字段。
因此，原来的随机模板选择实际让随机种子同时改变了方法协议，而不仅是控制搜索随机性。

本接口现枚举固定依赖版本提供的 TAP 模板，规范化既有占位符别名后，只接受唯一一条显式声明
`improvement` 与 `prompt` 键的模板。没有匹配或出现多个不同匹配时明确失败；不按 seed、长度、
实验结果或返回内容动态选择。所选模板全文不写入终端，完成元数据只记录长度、SHA256 和策略。
这使三个 seed 使用同一模板，同时保留其余随机过程。根据已记录的运行指纹，seeds100/200
已经使用该模板，因此保留既有完成结果；只需重跑零目标调用、零响应的 seed300。

三方法均调用原 LocalLLM；外部框架的目标模型类仅用于类型兼容和调用转接，
不再另加载一个模型或另外选择目标会话模板。多消息目标输入若不兼容会明确失败，
不会静默拼接为字符串。

## 安装与无调用检查

先完整解压原源码包到一个新目录，再将此增量包解压到同一目录。
不要覆盖正在运行的工程；不要同时启动旧工程 GPU 实验。

```bash
source .venv/bin/activate
cd <repository-root>
python -m unittest discover -s extensions_q20 -p 'test_*.py'
python extensions_q20/run.py audit
python extensions_q20/run.py plan
```

模型/API 配置继续从你已有的私有配置加载；不把密钥写入本目录：

```bash
set -a
source .env
set +a
export EASYJAILBREAK_ROOT=/path/to/EasyJailbreak
python extensions_q20/run.py audit --runtime
```

无 API、无 GPU 的边界检查通过后才允许 `run`。外部依赖源码或桥接代码发生变化时，
检查戳失效，需要重新检查；不会自动升级或重装依赖。

模型路径读取 `TARGET_LLAMA`、`TARGET_QWEN15`、`TARGET_QWEN3`、`TARGET_QWEN`、`TARGET_VICUNA`。
新增外部方法使用 MUTATION_* 服务作为方法内部的生成与评估模型；统一离线评估使用 RACOFUZZ_*。
两者模型与服务须按实验协议一致配置。

## 首个正式条件：单独验证，不重复调用已完成单元

```bash
python extensions_q20/run.py run --method tap --dataset gptfuzzer --model llama32_3b --seed 100
python extensions_q20/run.py evaluate --method tap --dataset gptfuzzer --model llama32_3b --seed 100
```

`GENERATION_PASS_EVALUATION_PENDING` 只表示生成阶段完成，不能当作完整实验通过。
`LEGACY_EVALUATION_COMPLETENESS_PASS` 只说明旧评估入口完成了全部记录；
它不证明旧 Judge 的输出解析不存在静默降级。原 Judge 已知有失败回退/标签转换行为，
本增量包未修改它，不应把这个标志称作“强可靠性审计通过”。
需要改变旧 Judge 行为时必须单独列出修改和重新评估范围，而非偷偷改核心。

### TAP 统计计数兼容修复

首次服务器运行在 3 条响应落盘后，因统计打印读取
`self.evaluator.eval_model.generate.calls` 而失败。
适配层现仅在 TAP 所用接口缺少该属性时增加调用计数，保留已有计数器。
参数、返回值、异常继续原样传递；计数是 Python 方法调用次数，不是 API 底层重试次数，
也不是成功次数，不能用它替代 Recorder 的实际目标预算记录。
`audit --runtime` 现额外验证实际 evaluator 对象上的计数读取和更新，但仍不是完整运行验证。
更新桥接代码后旧接口检查戳失效，必须重新运行 `audit --runtime`。
此前失败的 3 条记录须单独归档，不能与重跑结果拼接；本版本没有恢复方法内部搜索状态的功能。

`--method` 还支持 renellm、deepinception、ra_cofuzz、strict_gptfuzzer、pair。
`--dataset` 支持 gptfuzzer、advbench、jailbreakbench；
`--model` 支持 llama32_3b、qwen25_1_5b、qwen25_3b、qwen25_7b、vicuna_7b。
现已补齐 AdvBench、JailbreakBench 的 `*_q20_seed1234_with_targets.csv`，
TAP/PAIR 可读取三个数据集的参考列。两份新增文件各 20 行，保留原问题和顺序，
按问题文本完全相等匹配，不模糊匹配、不生成回答、不填统一占位文本。
新增 `target` 是配套数据提供的辅助响应前缀，不是完整标准答案或评估标签。
这次补全不代表外部方法的 GPU/API 运行已经验证。

来源为固定版本的公开配套数据：

- AdvBench：[llm-attacks 数据文件](https://github.com/llm-attacks/llm-attacks/blob/098262edf85f807224e70ecd87b9d83716bf6b73/data/advbench/harmful_behaviors.csv)，字段 `goal` / `target`。
- JailbreakBench：[JBB-Behaviors 数据文件](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors/blob/886acc352a31533ffbcf4ef22c744658688086fc/data/harmful-behaviors.csv)，字段 `Goal` / `Target`。

`reference_provenance.json` 记录来源版本、源文件哈希、20 行对应位置和每行指纹；
启动审计检查三个数据集参考列，新补齐的两份文件还检查来源记录与文件一致性。
日常检查完全离线，运行不会重新下载数据，也不会修改原文件：

```bash
python extensions_q20/reference_data.py
```

仅在新增文件缺失、需要从固定来源重建时，才使用 `--restore`。
该命令需要联网，遇到已存在且内容不同的文件会失败并保留文件，不覆盖数据。
旧扩展包使用统一占位文本的结果不能直接视为采用本参考列的新协议结果。

## 保存、失败和历史结果

### TAP 空候选兼容版本

服务器 seed300 在目标调用 0 次时失败：原生成器的全部分支未得到解析器接受的候选，
筛选器仍直接访问 dataset[0]；另一个边界是仅有一个候选且未通过筛选时，原回退访问第二项。
本版本只在接口层处理，不写入 EasyJailbreak 依赖目录、不修改提示文本或原解析器文件。

- 保留原解析器优先：原解析成功时逐字保留其返回值；原解析失败后，才扫描忽略引号内括号的
  完整平衡字典块，依次尝试标准 JSON 和 Python 字面量。兼容结果仍必须是字典、包含
  `improvement`，并含非空字符串 `prompt`；不补写、不改写字段值。
- 无完整括号、缺少必需字段、非字符串或空 `prompt` 仍明确拒绝；不会从自由文本生成候选。
  安装时检查服务器原解析器的固定源码结构，结构漂移则停止。完成元数据分别记录原解析接受、
  兼容解析接受和拒绝次数。
- 正常批次直接使用原生成器；两个及以上候选的筛选直接使用原筛选器。
- 仅原生成器返回空批次时，每个问题最多允许额外恢复一个变异批次。恢复沿用同一批次
  在首次尝试结束后的原对象状态，不复制动态 Instance、不重设 RNG、不更换 seed；
  分支内部的原重试上限不变。
- 再次为空或同题恢复额度已耗尽时，明确报生成异常，保留已写记录，不生成完成标记。
  不跳过问题、不把异常记为普通评估标签，不返回空集合给后续 max()。
- 单候选筛选只在内存中将“回退取前两项”改为“回退取实际存在的前至多两项”。
  不复制候选、不额外评分；源函数结构不符合预期时停止安装/审计。
- 目标预算仍为每题 10 次、每单元 200 次；额外变异调用可能增加 API 成本。
  tap_compat_events.json 与完成元数据记录恢复事件和逻辑 generate 调用计数；
  此计数不包含 SDK 内部重试，也不等于目标调用数。失败时同样保留事件文件。
- 既有成功单元不自动重跑、旧完成标记不改写。汇总时保留其源码指纹和新兼容版本标识，
  如报告总 API 成本，必须计入恢复调用，不能仅报告目标预算。

这是一项明确记录的可靠性策略变更，不将其描述为源代码完全未变。
适配器或两个辅助模块变化后须重跑 audit --runtime；检查戳包含这些文件指纹。

新结果独立放在 `extension_outputs/方法/数据集/模型/seed编号/`。
生成与评估分别写完成标记；仅在指纹和结果哈希一致时跳过重复调用。
失败目录保留，不自动覆盖；确认后将该精确单元目录移动到备份目录再重试。
锁只覆盖本增量工程，不能阻止你在其他工程手动启动 GPU 工作。

原有主比较、三分支消融、已有跨数据集/模型及统计分析均不默认重跑。
旧结果与新实验不自动合并：原版运行种子、API 服务时间和部分规则与此前方案不同，
必须在统计表里分开标注，不能只凭文件存在认定可直接做同协议比较。

初始增量交付只包含主比较接口；下节记录随后加入并单独验证的选择机制消融接口。

## 选择机制消融（2026-09-06 增量）

主比较的 Full RA-CoFuzz 三个种子已经完成，本阶段不重复运行 Full。新增三个互斥的
选择分支消融，每个变体只将一个分支比例置零，并按冻结协议重新分配剩余比例：

| 变体 | Warm-up | MCTS | DP | Elite |
|---|---:|---:|---:|---:|
| `hybrid_wo_elite` | 2 | 0.50 | 0.50 | 0.00 |
| `hybrid_wo_dp` | 2 | 0.80 | 0.00 | 0.20 |
| `hybrid_wo_mcts` | 0 | 0.00 | 0.70 | 0.30 |

`hybrid_wo_mcts` 同时将 warm-up 设为0，避免前两步仍通过 MCTS。其余 Full 参数保持：
Q20、B200、Llama-3.2-3B-Instruct、seeds 100/200/300、Greedy、128 tokens、DP alpha=0.5、
epsilon=0.10、DP top-k=5、Elite top-k=5、RA 模式与在线强评开启。

原选择器的通用回退链可能在某个分支比例为0时仍调用该分支。为保证 `w/o` 的含义严格成立，
本接口仅在消融单元运行期间临时约束 `select()`：被移除分支既不参加常规抽样，也不参加回退；
其他分支沿用原选择函数和打分函数。上下文结束后立即恢复原方法，因此不改变 Full 或其他实验。
完成标记记录每个分支的尝试数、实际选择数、回退次数和总步数；最终审计要求被移除分支的
尝试数与选择数均为0，并要求分支选择总数等于总选择步数。

消融只允许 `ra_cofuzz + gptfuzzer + llama32_3b`，其他方法、数据集或模型会明确拒绝，
避免误把后续泛化实验混入机制消融。输出独立保存在：

```text
extension_outputs/selection_ablations/<variant>/gptfuzzer/llama32_3b/seed<seed>/
```

查看固定的9单元计划：

```bash
python extensions_q20/run.py plan --suite selection_ablations
```

服务器加载既有私有配置并完成只读检查后，使用队列脚本依次运行：

```bash
bash extensions_q20/run_selection_ablations.sh
```

每个单元分别生成完成标记和评估完成标记；已有且指纹一致的单元会跳过，失败或部分完成
的目录不会自动覆盖。全部完成后脚本自动执行安全汇总审计。最终成功标志为：

```text
Q20_SELECTION_ABLATION_9_CELLS_AUDIT_PASS
Q20_SELECTION_ABLATIONS_WORKFLOW_FINISHED
```

审计仅打印变体、seed、计数和聚合指标，不打印问题、候选内容或模型响应。

## 奖励与在线反馈消融（2026-09-06 增量）

在主比较18单元和选择分支消融9单元均完成后，本阶段新增三种互斥处理，仍固定为
GPTFuzzer-Q20、Llama-3.2-3B-Instruct、B200、128 tokens、seeds 100/200/300：

| 变体 | 在线 StrongJudge | Label-1 搜索奖励 | 启发式 partial 奖励 |
|---|---|---|---|
| `wo_label1_reward` | 原选择性调用 | 关闭 | 保持 |
| `wo_online_strongjudge_feedback` | 关闭（必须为0次） | 无在线标签 | 保持 |
| `full_online_review` | 每条候选调用（必须为200次） | 保持 | 保持 |

`wo_label1_reward` 只把在线 Label-1 在 fitness、种子保留和 Hybrid/MCTS/DP/Elite 父模板
选择中的有效标签映射为未判别；原始 Label-1 仍写入运行记录并进入统计。它不关闭
`partial_compliance_flag`、clear-success proxy 等独立启发式信号，因此不得描述为
“w/o all partial reward”。此处理复现历史版本的 `GPTFUZZ_DISABLE_LABEL1_REWARD` 语义，
但通过运行时上下文临时安装并在单元结束后恢复，不改写上传的原源码文件。

`wo_online_strongjudge_feedback` 仅关闭搜索环内 StrongJudge，启发式评估、Hybrid比例、
生成预算及最终离线全量复评不变。`full_online_review` 将每条候选都送入在线判别；完成审计
要求逻辑在线调用数精确等于200。SDK内部网络重试不计入该逻辑调用数，API账单仍应以服务商
记录为准。三种变体最终都对全部200条响应执行同一离线 StrongJudge 和 RoBERTa 复评。

输出独立保存在：

```text
extension_outputs/feedback_ablations/<variant>/gptfuzzer/llama32_3b/seed<seed>/
```

查看固定的9单元计划：

```bash
python extensions_q20/run.py plan --suite feedback_ablations
```

运行队列：

```bash
bash extensions_q20/run_feedback_ablations.sh
```

脚本逐单元生成、评估并核对处理是否真正生效，同时复用既有 Full RA-CoFuzz 三个种子作为
参照而不重跑。最终成功标志为：

```text
Q20_FEEDBACK_ABLATION_9_CELLS_AUDIT_PASS
Q20_FEEDBACK_ABLATIONS_WORKFLOW_FINISHED
```

## 跨数据集泛化实验（2026-09-06 增量）

完成 GPTFuzzer-Q20 主比较及两组消融后，本阶段保持模型与算法参数不变，仅切换问题集，验证
结论能否泛化到 AdvBench-Q20 和 JailbreakBench-Q20。每个数据集均使用六种方法和
seeds 100/200/300，共36个新单元：

| 阶段 | 方法 | 数据集 | 单元数 |
|---|---|---|---:|
| Core | RA-CoFuzz、Strict GPTFuzzer、PAIR | AdvBench、JailbreakBench | 18 |
| Remaining | TAP、ReNeLLM、DeepInception | AdvBench、JailbreakBench | 18 |

Core 阶段包含完整方法、同源严格基线和主实验中最稳定的外部基线，优先形成论文核心泛化
对照。Remaining 阶段补齐其余三个外部方法。完整队列会在 Core 18单元结束后执行强制审计，
审计通过才继续后18单元。

冻结设置为 Q20、每题目标调用上限10、单元总上限B200、Llama-3.2-3B-Instruct、128 tokens及
seeds 100/200/300。RA-CoFuzz和Strict GPTFuzzer必须完整记录200条；PAIR、TAP与
ReNeLLM允许按原算法提前结束，但必须覆盖20题且每题不超过10条；DeepInception按其原生
单次生成流程固定为20条。因此跨方法同时报告问题级成功率、响应级成功率及实际调用量，
不能只比较响应总数。

PAIR和TAP读取已审计的参考列；两份新数据集均已通过20/20精确来源匹配。所有单元仍执行
相同的全量离线 StrongJudge 与 RoBERTa 评估，不复用历史旧实验数字。

查看矩阵：

```bash
python extensions_q20/run.py plan --suite cross_dataset_core
python extensions_q20/run.py plan --suite cross_dataset_all
```

运行完整队列：

```bash
bash extensions_q20/run_cross_dataset.sh all
```

也可只运行核心阶段：

```bash
bash extensions_q20/run_cross_dataset.sh core
```

全部成功标志：

```text
Q20_CROSS_DATASET_CORE_18_CELLS_AUDIT_PASS
Q20_CROSS_DATASET_CORE_18_CELLS_FINISHED
Q20_CROSS_DATASET_ALL_36_CELLS_AUDIT_PASS
Q20_CROSS_DATASET_ALL_36_CELLS_FINISHED
```

审计验证每个文件、结果哈希、实际响应数、20题覆盖、单题上限、离线标签完整性和运行签名，
只输出安全状态与聚合指标。

## 跨模型泛化实验（2026-09-06 增量）

本阶段固定 GPTFuzzer-Q20、B200、128 tokens、seeds 100/200/300 和六种方法，只切换
目标模型。已完成的 Llama-3.2-3B-Instruct 主比较18单元作为参考，不重新运行。新增模型为
Qwen2.5-1.5B-Instruct、Qwen2.5-3B-Instruct、Qwen2.5-7B-Instruct 与 Vicuna-7B-v1.5，共72个新单元：

| 阶段 | 方法 | 新模型数 | 单元数 |
|---|---|---:|---:|
| Core | RA-CoFuzz、Strict GPTFuzzer、PAIR | 4 | 36 |
| Remaining | TAP、ReNeLLM、DeepInception | 4 | 36 |

完整队列先完成 Core 36单元并审计，通过后才进入 Remaining。不同模型仍遵守同一预算、
覆盖率和离线评估规则；原生提前停止方法按实际响应数报告。模型对应关系由冻结协议读取：
`TARGET_QWEN15`、`TARGET_QWEN3`、`TARGET_QWEN`、`TARGET_VICUNA`。目标路径和原始文本
不会打印到聚合审计日志。

查看矩阵：

```bash
python extensions_q20/run.py plan --suite cross_model_core
python extensions_q20/run.py plan --suite cross_model_all
```

运行完整队列：

```bash
bash extensions_q20/run_cross_model.sh all
```

全部成功标志：

```text
Q20_CROSS_MODEL_CORE_36_CELLS_AUDIT_PASS
Q20_CROSS_MODEL_CORE_36_CELLS_FINISHED
Q20_CROSS_MODEL_ALL_72_CELLS_AUDIT_PASS
Q20_CROSS_MODEL_ALL_72_CELLS_FINISHED
```

### 空目标响应的离线评估规则（2026-09-07）

目标模型调用成功但返回长度为0的字符串时，该次调用仍计入目标预算，不重试目标模型，也不
调用离线StrongJudge。统一将其记为Label-0“无有效内容”，并在完成标记和汇总中单独记录
`empty_response_policy`与数量。此规则只接受原评估器产生的精确
`missing prompt or response`记录；若提示同时缺失、行未对齐、存在其他异常或汇总计数不吻合，
则明确失败，不把其他错误转换成Label-0。

Vicuna-7B-v1.5、RA-CoFuzz、seed100首次评估发现15/200条空响应。生成文件和200次目标调用保持
不变，仅归档不完整的四个离线评估文件并重新执行评估。由于修复后扩展指纹变化，Core队列
不能从头重启；使用专用断点脚本补评当前单元，再执行尚未开始的8个Vicuna核心单元：

```bash
bash extensions_q20/resume_cross_model_core_vicuna.sh
```

该脚本不会访问或重跑此前已经完成的27个Qwen核心单元。Core审计通过后，再使用原
`run_cross_model.sh remaining`补齐Remaining 36单元。
