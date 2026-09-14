# FedCompass · 多模态联邦计算平台

FedCompass 是一个面向**多模态联邦学习**研究与演示的一体化平台，覆盖从实验配置、联邦训练、安全攻防到结果分析的全流程。

平台以**插件化**的方式组织联邦算法、优化器、攻击与防御方法，并通过**四个协同智能体**在实验运行时提供运行保障、安全监控与结果解读。前端是一个零构建的静态单页应用，后端为 FastAPI 服务，可一键 Docker 部署。

---

## 功能特性

- **插件化联邦算法库** — 68 个算法条目，涵盖 FedAvg/FedProx/FedDyn/SCAFFOLD 等经典基线、实验室自研方法（FedCVC、FedDTC、Co-Evo、FedULIP 等）以及 ReID / 3D 多模态方向的对比方法
- **安全攻防** — 9 个防御方法（Krum、Multi-Krum、Trimmed-Mean、Median、FLTrust、FLAME、FLDetector、FLBeeline、VERT）与 7 个攻击条目（特征投毒、原型攻击、高斯噪声、模型替换、AGR、ALIE）
- **多智能体协同** — 实验场景、运行保障、安全监控、结果分析四个智能体贯穿实验全生命周期
- **实时运行监控** — 通过 SSE 推送轮次指标、客户端参与情况、安全观测与智能体证据卡片
- **多模态任务** — 图像分类（MNIST/CIFAR-10）、行人重识别（Market-1501）、3D 点云（ModelNet40）
- **本地大模型** — 内置 llama.cpp + Qwen3-0.6B（GGUF Q8_0），无需外部 API 即可运行智能体；也可按请求切换到外部 OpenAI 兼容服务
- **零构建前端** — 原生 HTML/CSS/JS，后端不可用时自动回退到本地 mock 数据

---

## 系统架构

```
┌──────────────────────────────────────────────────────────┐
│  前端 (原生 JS 静态页)                                      │
│  总览 · 模块能力 · 实验配置 · 运行监控 · 结果对比 · 应用场景    │
└───────────────────────┬──────────────────────────────────┘
                        │ REST + SSE
┌───────────────────────▼──────────────────────────────────┐
│  后端 (FastAPI)                                           │
│  ├─ API 层     overview / modules / algorithms / experiments│
│  │             results / events / resources / workflow     │
│  ├─ 编排层     orchestrator (多智能体工作流 + 工具注册表)     │
│  ├─ 智能体层   agents (运行保障 / 安全监控 / 结果分析)         │
│  │             agent  (实验场景 / 配置助手)                   │
│  ├─ 插件层     plugins (算法 / 优化器 / 攻击 / 防御 / 应用)    │
│  ├─ Runner     runners (mock / fedcompass / 真实联邦任务)     │
│  └─ 存储       SQLAlchemy(async) + SQLite                  │
└───────────────────────┬──────────────────────────────────┘
                        │ OpenAI 兼容 /chat/completions
┌───────────────────────▼──────────────────────────────────┐
│  模型服务 (llama.cpp) — Qwen3-0.6B Q8_0 GGUF，端口 8001      │
└──────────────────────────────────────────────────────────┘
```

---

## 目录结构

```
.
├── 0623v1李-后端/                  # FastAPI 后端
│   ├── app/
│   │   ├── api/                    # 路由层
│   │   ├── agents/                 # 运行保障 / 安全监控 / 结果分析智能体
│   │   ├── agent/                  # 实验场景智能体（配置助手）
│   │   ├── orchestrator/           # 多智能体编排与工具注册表
│   │   ├── plugins/                # 算法 / 优化器 / 攻击 / 防御 / 应用插件
│   │   │   ├── algorithms/         #   联邦算法实现
│   │   │   ├── defenses/           #   防御方法
│   │   │   ├── attacks/            #   攻击方法
│   │   │   ├── optimizers/         #   SAM / ESAM / HSAM
│   │   │   ├── apps/               #   任务级插件（reid / ulip3d / vision）
│   │   │   ├── registry.py         #   插件注册表（ID → 实现）
│   │   │   └── base.py             #   ClientState / RoundResult 数据结构
│   │   ├── runners/                # 实验执行器
│   │   ├── core/                   # 配置与数据库
│   │   ├── models/ schemas/        # ORM 模型与 Pydantic 校验
│   │   └── seed/                   # 模块/算法/参考文献目录种子数据
│   ├── tests/                      # unittest 测试（含算法冒烟与多智能体安全测试）
│   └── requirements.txt
│
├── 0624v1冠-前端/FL系统/frontend/   # 前端静态原型（零构建）
│   ├── index.html  app.js  styles.css
│   └── samples/                    # 界面用示例图
│
├── docker/                         # 三个服务的构建文件
│   ├── backend/Dockerfile
│   ├── frontend/Dockerfile  nginx.conf
│   └── model/Dockerfile            # llama.cpp + Qwen3-0.6B
├── scripts/                        # 数据集准备与数据回填脚本
├── docker-compose.yml
├── 部署说明.md                      # 详细部署文档
├── 启动方式.txt                     # 本地启动速查
├── 前后端字段链路对照表.md
└── 实验场景智能体模型接入说明.md
```

---

## ⚠️ 关于未纳入版本控制的大文件

为保证仓库体积可控，以下内容**已通过 `.gitignore` 排除，不在本仓库中**。克隆后如需完整功能，请自行准备：

| 路径 | 体积 | 用途 | 获取方式 |
|---|---|---|---|
| `models/qwen3-0.6b/Qwen3-0.6B-Q8_0.gguf` | ~610 MB | 本地智能体模型 | 从 HuggingFace 下载 Qwen3-0.6B 的 Q8_0 GGUF 量化版本 |
| `real_data/` | ~205 MB | 真实数据集 | 见下方「数据集准备」 |
| `.venv-fedcompass/` | ~662 MB | 本地虚拟环境 | 按下方步骤自行创建 |

> **说明**：`.dockerignore` **特意没有**排除 `models/` 和 `*.gguf`——构建 `model` 镜像时必须用到该权重文件。

**数据集准备**：`real_data/` 下包含 CIFAR-10、ModelNet40-mini、Market1501-mini。仓库提供了两个准备脚本：

```bash
python scripts/prepare_modelnet40_mini.py
python scripts/prepare_market1501_mini.py
```

CIFAR-10 可通过 `torchvision.datasets.CIFAR10(download=True)` 获取。

**降级运行**：缺少 `real_data/` 时后端仍可正常启动，目录数据（模块/算法/数据集/参考文献）与演示实验均可用，但真实数据实验不可用。

---

## 快速开始

### 方式一：Docker 部署（推荐）

**前置条件**：Docker Engine + Docker Compose v2，且 `models/qwen3-0.6b/Qwen3-0.6B-Q8_0.gguf` 已就位。

```bash
docker compose build
docker compose up -d
```

国内网络如访问官方源较慢，可在构建时临时指定镜像源：

```bash
docker compose build \
  --build-arg DEBIAN_MIRROR=mirrors.aliyun.com \
  --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
```

访问 **http://服务器IP:8080** ，健康检查：

```bash
curl http://服务器IP:8080/healthz     # 期望返回 {"status":"ok"}
```

Compose 会启动三个服务：`model`（llama.cpp，容器内 8001）、`backend`（FastAPI，容器内 8000）、`frontend`（nginx，映射到宿主机 8080）。SQLite 数据存放在 Docker volume `fedcompass_db` 中。

> ⚠️ 不要执行 `docker compose down -v`，它会删除数据库卷。

### 方式二：本地运行

**前置条件**：Python 3.13（开发环境）。后端依赖见 `0623v1李-后端/requirements.txt`。

```bash
# 1. 创建虚拟环境并安装依赖
python3 -m venv .venv-fedcompass
.venv-fedcompass/bin/pip install -r 0623v1李-后端/requirements.txt

# 2. 启动后端（终端 1）
cd 0623v1李-后端
../.venv-fedcompass/bin/python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

# 3. 启动前端（终端 2，回到仓库根目录后执行）
cd 0624v1冠-前端/FL系统/frontend
python3 -m http.server 4173
```

访问 **http://127.0.0.1:4173** ，后端 API 文档在 **http://127.0.0.1:8000/docs**。

**首次启动**会自动写入 4 个演示实验（每个 100 轮指标与事件），便于直接查看运行监控与结果对比页面。该填充仅在实验表为空时执行，重启不会重复写入，也不会覆盖你创建的真实实验。

**前端后端地址**由 `app.js` 自动推导，无需改代码：本地静态服务（端口 4173）指向 `http://localhost:8000`；Docker 部署走 nginx 同源反代。如需指向其他地址，在 `index.html` 中取消 `window.__FEDCOMPASS_API_BASE__` 那行的注释并填入地址即可。

> 后端不可达时，前端会自动回退到内置模拟数据并在侧边栏提示，不会白屏。注意：接口返回**空列表**时前端如实显示为空，不会用模拟数据填充。

**本机跑智能体（可选）**：若希望在本机而非 Docker 中运行 Qwen 模型服务，可用 llama.cpp 在 8001 端口启动：

```bash
llama-server --model models/qwen3-0.6b/Qwen3-0.6B-Q8_0.gguf \
  --alias Qwen/Qwen3-0.6B --host 0.0.0.0 --port 8001 \
  --ctx-size 8192 --gpu-layers 0 --reasoning off --jinja
```

---

## 插件体系

所有插件通过 `app/plugins/registry.py` 中的注册表按 ID 映射到实现，新增方法只需实现对应函数并注册：

| 注册表 | 条目数 | 接口约定 |
|---|---|---|
| `ALGORITHM_REGISTRY` | 68 | `(aggregate_fn, local_train_fn)` |
| `DEFENSE_REGISTRY` | 9 | `(aggregate_fn, filter_fn)` |
| `ATTACK_REGISTRY` | 7 | 梯度变换函数 |

算法条目按来源分为几类：

- **核心基线** — `fedavg`、`fedprox`、`feddyn`、`scaffold`，以及 `fednova`/`fedadam`/`fedyogi`/`fedbn`/`ditto`/`pfedme`/`fedproto`/`feddf`/`fedkd`
- **实验室方法** — `fedcvc`、`feddtc`、`rfl_nlcp`、`gfed_hsam`、`fedcads`、`co_evo`、`fedulip`
- **优化器与论文基线** — `sgd`/`sam`/`esam`/`hsam` 及 `a_fedpd`、`fedspeed`、`fedsmoo`、`feddc`、`fedvra`、`fedtoga` 等
- **ReID 方向** — `moon`、`mixstyle`、`crossstyle`、`fedreid`、`fedpav`、`snr`、`dacs`、`sscu`
- **3D 多模态方向** — `ulip`、`pointclip`、`fedkgcoop`、`fedvpt`、`fedtpg`、`fedcocoop`、`fedmaple`、`fedclip`、`fedmvp`

> 防御方法与攻击条目同时也可作为「算法」下拉项被选中（用于压力测试），因此也出现在 `ALGORITHM_REGISTRY` 中。

核心数据结构定义在 `app/plugins/base.py`：

- `ClientState` — 客户端状态，含 SCAFFOLD 控制变量、FedDyn 历史梯度、FedDC 更新缓存、FedCVC 静默计数、信任分数等字段
- `RoundResult` — 单轮结果，含损失/精度、参与/静默/恶意客户端、防御检测结果、梯度漂移、通信开销

---

## 多智能体协同

平台包含四个智能体，分为「请求级」与「后台运行级」两类：

| 智能体 | 触发方式 | 职责 |
|---|---|---|
| **实验场景智能体** | 前端请求级 | 自然语言 → 实验配置；支持「本地模型 / API 模型」按请求切换 |
| **运行保障智能体** | 后台每 3 轮 | 检查运行状态，输出证据与建议动作 |
| **安全监控智能体** | 后台每 3 轮 | 构建轮次安全快照，识别异常客户端 |
| **结果分析智能体** | 实验结束后 | 生成观察、可支持结论、不能推出的结论、限制与补充实验建议 |

- 后台三个智能体固定使用本地 Qwen 服务（`AGENT_LLM_*` 配置），**不受**实验场景智能体的请求级模型选择影响
- 运行中检查周期由 `RUNTIME_AGENT_CHECK_INTERVAL_ROUNDS` 控制（默认 3），**最后一轮无论是否整除都会检查**
- 本地模型不可用时，配置助手自动回退到确定性的本地规则解析

---

## API 接口

后端主要路由（完整定义见 `0623v1李-后端/app/api/`，交互式文档 `/docs`）：

| 分组 | 示例端点 |
|---|---|
| 总览 | `GET /api/overview`、`GET /api/overview/capability-map` |
| 模块与资源 | `GET /api/modules`、`GET /api/algorithms`、`GET /api/datasets`、`GET /api/optimizers`、`GET /api/defenses`、`GET /api/attacks` |
| 实验 | `POST /api/experiments`、`POST /api/experiments/{id}/preflight`、`POST /api/experiments/{id}/start`、`POST /api/experiments/{id}/stop` |
| 监控 | `GET /api/experiments/{id}/metrics`、`GET /api/experiments/{id}/events/stream`（SSE） |
| 智能体 | `GET /api/experiments/{id}/workflow`、`GET /api/experiments/{id}/operations-report`、`GET /api/experiments/{id}/security-report`、`GET /api/experiments/{id}/result-analysis` |
| 结果 | `GET /api/results`、`POST /api/results/compare`、`GET /api/results/export` |

---

## 配置项

后端配置位于 `0623v1李-后端/app/core/config.py`，通过环境变量或 `.env` 覆盖。可参考 `.env.example` 模板：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `DATABASE_URL` | `sqlite+aiosqlite:///./fedcompass.db` | 数据库连接串 |
| `REAL_DATA_ROOT` | *（空，回退到仓库根 `real_data/`）* | 真实数据集根目录 |
| `CORS_ORIGINS` | 含 localhost 多个端口 | 允许的跨域来源 |
| `LLM_PROVIDER` | `api` | 实验场景智能体的模型来源：`api` / `local` |
| `LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL` | *（空）* | 外部 OpenAI 兼容服务 |
| `LLM_LOCAL_BASE_URL` / `LLM_LOCAL_MODEL` | `http://127.0.0.1:8001/v1` / `Qwen/Qwen3-0.6B` | 本地模型服务 |
| `AGENT_LLM_PROVIDER` | `local` | 后台三个智能体的模型来源 |
| `AGENT_LLM_TIMEOUT_SECONDS` | `180` | 后台智能体超时（CPU 推理较慢，故上限较高） |
| `RUNTIME_AGENT_CHECK_INTERVAL_ROUNDS` | `3` | 运行中检查间隔轮数 |

> 🔐 **切勿把 API Key 写入仓库**。前端的 API Key 仅随当前请求传递，不写入数据库、浏览器存储或响应。

---

## 运行测试

测试基于 Python 标准库 `unittest`（异步用例内部通过 `asyncio.run` 驱动），无需额外测试框架：

```bash
cd 0623v1李-后端
../.venv-fedcompass/bin/python -m unittest discover -s tests -v
```

覆盖范围：

- `test_algorithm_smoke.py` — 算法插件冒烟测试
- `test_config_assistant.py` — 配置助手
- `test_experiment_config_constraints.py` — 实验配置约束校验
- `test_multi_agent_safety.py` — 多智能体安全边界与工具调用约束
- `test_pytorch_reproduction_smoke.py` — PyTorch 复现冒烟测试

---

## 文档

| 文档 | 内容 |
|---|---|
| `部署说明.md` | Docker 部署、数据持久化、配置助手、数据库迁移 |
| `Docker部署构建计划.md` | 镜像构建方案与构建参数说明 |
| `启动方式.txt` | 本地前后端启动速查 |
| `前后端字段链路对照表.md` | 前后端字段映射 |
| `实验场景智能体模型接入说明.md` | 配置助手的模型接入方式 |
| `0624v1冠-前端/FL系统/FedCompass展示系统设计.md` | 前端页面与交互设计 |
| `0624v1冠-前端/FL系统/0623李-F&B_API_Doc.md` | 前后端接口约定 |

---

## 当前部署边界

- 当前方案为**单后端容器 + SQLite**，实验运行状态与 SSE 事件依赖单进程内存，**不建议直接扩展多个 backend 副本**
- 后端重启时，遗留的 `running` 实验会被标记为 `stopped`
- 模型容器按 CPU 模式启动（`--gpu-layers 0`），适合通用 Linux 服务器；如有 GPU，应另行构建匹配 CUDA/Metal 的模型镜像
- 后续需要高可用时，应先迁移 PostgreSQL 与共享任务队列，再扩展多副本
