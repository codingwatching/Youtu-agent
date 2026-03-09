# SWE-bench Evaluation

SWE-bench 评测流程：agent 在独立 Docker 沙箱中修复真实 GitHub issue，提取 patch，离线运行测试判定是否解决。

## Quick Start

```bash
# 1. 导入数据集
python scripts/data/process_swe_bench.py --subset verified --split test

# 2. Rollout（agent 在沙箱中修 bug + 提取 patch）
python scripts/run_eval.py --config_name swe_bench --exp_id swe_bench_verified-test01

# 3. 导出 patch
python scripts/data/export_swe_bench_patches.py --exp_id swe_bench_verified-test01

# 4. 离线评测（需要 Docker 环境）
python -m swebench.harness.run_evaluation \
  --dataset_name princeton-nlp/SWE-bench_Verified \
  --predictions_path patches.jsonl \
  --max_workers 4 \
  --run_id swe_bench_verified-test01
```

## 配置文件

### 环境配置 `configs/env/swerex.yaml`

定义 AGS 沙箱参数：镜像、资源规格、SWE-ReX 端口、存储挂载等。

```yaml
name: swerex
config:
  deployment_type: ags          # "ags" 或 "remote"
  image: ${oc.env:AGS_SANDBOXTOOL_IMAGE}  # SWE-bench 评测时会被 per-instance 覆盖
  cpu: "2"
  memory: "4Gi"
  timeout: "2h"
  workspace: /                  # agent 工作目录，评测时覆盖为 /testbed
  bash_timeout: 120             # 单条命令超时（秒）
```

关键点：`image` 和 `workspace` 在评测时由 `SWEBenchmark.rollout_one()` 按 instance 覆盖。

### Agent 配置 `configs/agents/swe/swe_bench.yaml`

```yaml
defaults:
  - /model/base@model
  - /env/swerex@env

agent:
  name: swe
  instructions: |
    You are a skilled software engineer...

env:
  name: swerex
  config:
    workspace: /repo
```

Agent 使用 SWERexEnv 提供的工具（bash, read_file, write_file, edit_file），无需额外配置 toolkits。

### 评测配置 `configs/eval/swe_bench.yaml`

```yaml
defaults:
  - /agents/swe/swe_bench@agent

exp_id: "swe_bench_verified"
data:
  dataset: "SWEBench_Verified"
concurrency: 3
```

## 代码实现

### SWERexEnv (`utu/env/swerex_env.py`)

AGS 沙箱环境，封装 SWE-ReX runtime。

```
SWERexEnv
├── build()          # 启动 AGS 沙箱 → 连接 runtime → 创建 bash session
│   ├── _build_ags()     # TencentAGSDeployment
│   └── _build_remote()  # RemoteDeployment（调试用）
├── get_tools()      # 返回 [bash, read_file, write_file, edit_file]
├── cleanup()        # 停止沙箱，释放资源
└── _run_bash() / _read_file() / _write_file() / _edit_file()
```

设计要点：
- **Post-startup 自动禁用 pager**：`export GIT_PAGER=cat PAGER=cat`，避免 git 命令在 pexpect 伪终端中阻塞
- **工具独立提供**：通过 `get_tools()` 暴露，不依赖框架的 toolkit 机制
- **_MAX_OUTPUT_CHARS = 50,000**：工具输出截断上限

### SWEBenchmark (`utu/eval/benchmarks/swe_benchmark.py`)

继承 `BaseBenchmark`，override `rollout_one()` 实现 per-instance 沙箱生命周期。

```
rollout_one(sample):
  1. agent_config = self.config.agent.model_copy(deep=True)   # 深拷贝，并发安全
  2. agent_config.env.config["image"] = sample.meta["image_name"]
     agent_config.env.config["workspace"] = "/testbed"
  3. agent = get_agent(agent_config)
     await agent.build(trace_id)
  4. try:
       result = await agent.run(sample.augmented_question)
       # 写文件 + read_file 提取 patch（绕过 pager 问题）
       await agent.env._run_bash("git add -A && git diff --cached > /tmp/model.patch")
       patch = await agent.env._read_file("/tmp/model.patch")
     finally:
       await agent.cleanup()          # 必须销毁沙箱
  5. sample.update(response=..., extracted_final_answer=patch, stage="rollout")
```

核心设计：
- **`model_copy(deep=True)`**：每个 instance 独立 config 副本，避免并发覆盖
- **`finally: cleanup()`**：AGS 沙箱必须显式销毁，否则资源泄漏
- **Patch 提取走文件 API**：`git diff --cached > file` + `read_file`，比直接 `git diff` 更可靠（避免 pexpect pager 阻塞和 buffer 溢出）

### SWEBenchProcesser (`utu/eval/processer/swe_bench.py`)

```
SWEBenchProcesser (name="SWEBench")
├── preprocess_one()      # 构造 augmented_question（注入 repo 名 + problem_statement）
├── judge_one()           # 跳过（SWE-bench 离线评测）
└── calculate_metrics()   # 统计 patch_rate（有 patch 的比例）
```

### 数据导入 (`scripts/data/process_swe_bench.py`)

从 HuggingFace 加载 SWE-bench 数据集，写入 `DatasetSample` 表。

```
DatasetSample(
  dataset="SWEBench_Verified",
  source="SWEBench",
  question=instance["problem_statement"],
  meta={
    "instance_id": "astropy__astropy-12907",
    "repo": "astropy/astropy",
    "base_commit": "abc123",
    "image_name": "swebenchdocker.tencentcloudcr.com/swebench/sweb.eval.x86_64.astropy__astropy-12907:latest",
    "FAIL_TO_PASS": "...",
    "PASS_TO_PASS": "...",
    "patch": "...",           # gold patch
  },
)
```

镜像名规则：`sweb.eval.x86_64.{instance_id_escaped}:latest`，其中 `__` → `_1776_`，全小写。

### Patch 导出 (`scripts/data/export_swe_bench_patches.py`)

从 DB 读取 rollout 结果，生成 `swebench.harness.run_evaluation` 所需的 JSONL：

```json
{"instance_id": "astropy__astropy-13033", "model_name_or_path": "utu-agent", "model_patch": "diff --git a/..."}
```

## 架构总览

```
process_swe_bench.py          scripts/run_eval.py              export_swe_bench_patches.py
      │                              │                                   │
      │ HuggingFace                  │ Hydra config                      │ DB query
      ▼                              ▼                                   ▼
 ┌──────────┐    ┌─────────────────────────────────┐    ┌──────────────────────┐
 │DatasetSample│──>│ SWEBenchmark.rollout_one()       │──>│ patches.jsonl        │
 │  (DB)       │   │  ├── deep copy config            │   │  (JSONL)             │
 └──────────┘   │  ├── inject image + workspace      │   └──────────────────────┘
                │  ├── build agent (AGS sandbox)     │             │
                │  ├── agent.run() (fix bug)         │             ▼
                │  ├── git diff > file + read_file   │   swebench.harness
                │  └── cleanup (destroy sandbox)     │   .run_evaluation
                └─────────────────────────────────┘   (Docker, 离线)
```

## 环境变量

评测相关（在 `.env` 中配置）：

| 变量 | 说明 |
|------|------|
| `AGS_SANDBOXTOOL_ID` | AGS SandboxTool ID |
| `AGS_SANDBOXTOOL_IMAGE` | 默认容器镜像（SWE-bench 时被覆盖） |
| `AGS_SANDBOXTOOL_ROLEARN` | AGS 角色 ARN |
| `AGS_MOUNT_IMAGE_ID` | SWE-ReX 二进制挂载镜像 |
| `TENCENTCLOUD_SECRET_ID` | 腾讯云 Secret ID |
| `TENCENTCLOUD_SECRET_KEY` | 腾讯云 Secret Key |

## 注意事项

- SWE-bench 每个 instance 使用**不同的 Docker 镜像**（预装了对应仓库代码），由 `SWEBenchmark` 在 rollout 时动态注入
- 离线评测需要 Docker 环境（本地 macOS 无 Docker 时可在远程服务器运行）
- `--subset` 支持 `verified`（500 条）、`lite`、`full`
- `--concurrency` 控制并行沙箱数量，注意 AGS 资源配额
