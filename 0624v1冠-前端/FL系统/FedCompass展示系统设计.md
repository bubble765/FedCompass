# FedCompass 展示系统前后端设计

## 1. 系统定位

FedCompass 展示系统不是普通宣传页，而是一个可交互的联邦学习算法展示与实验管理控制台。第一屏应直接进入系统总览，展示系统模块、算法能力、实验状态、关键指标和典型应用场景。

核心目标：

| 目标 | 说明 |
| --- | --- |
| 模块展示 | 清晰展示 FedCore、FedAdapt、FedOpt、FedDistill、FedGuard、FLBeeline、FeaturePoison Attack、FedVision、FedReID、Fed3D、FedUtil 的职责。 |
| 算法匹配 | 每个模块能看到对应算法、可选基线、适用问题和关键机制。 |
| 实验演示 | 支持选择数据集、算法、异构设置、参与率、防御策略并启动演示实验。 |
| 过程监控 | 展示训练轮次、客户端参与、精度、loss、通信量、攻击检测结果等动态指标。 |
| 结果对比 | 支持不同算法、不同场景、不同攻击/防御组合的结果对比。 |
| 应用扩展 | 展示 Vision、ReID、3D 多模态三个应用方向的样例结果。 |

## 2. 推荐技术栈

### 前端

| 层级 | 推荐方案 | 说明 |
| --- | --- | --- |
| 框架 | React + TypeScript + Vite | 启动快，适合实验展示系统和后续工程化。 |
| 路由 | React Router | 按 Dashboard、Modules、Experiments、Results、Apps 拆页。 |
| 请求 | TanStack Query | 管理 API 请求、缓存、轮询和错误状态。 |
| 状态 | Zustand | 保存当前实验配置、筛选条件、全局 UI 状态。 |
| 图表 | ECharts 或 Recharts | 训练曲线、雷达图、柱状对比图、客户端热力图。 |
| UI | Ant Design 或 shadcn/ui | 建议选择偏工作台风格的组件库，保持密度适中、易扫描。 |
| 实时更新 | Server-Sent Events 或 WebSocket | 推送训练日志、轮次指标和任务状态。 |

### 后端

| 层级 | 推荐方案 | 说明 |
| --- | --- | --- |
| API 服务 | FastAPI | 类型清晰，便于生成 OpenAPI 文档。 |
| 数据库 | SQLite 起步，PostgreSQL 生产 | 演示系统可先用 SQLite，后续无缝迁移。 |
| ORM | SQLAlchemy + Alembic | 管理实验、算法、模块、指标、任务记录。 |
| 后台任务 | Celery/RQ 或 FastAPI BackgroundTasks | 运行实验、解析日志、生成图表和结果表。 |
| 实时通道 | SSE 或 WebSocket | 推送 experiment status、metric event、log event。 |
| 实验引擎 | Python plugin registry | 调用 fedcompass/algorithms、optimizers、defenses、apps。 |
| 文件存储 | 本地 outputs/，后续可换对象存储 | 保存日志、checkpoint、result.json、图表数据。 |

## 2.5 参考论文与代码资产映射

展示系统需要把每个模块的算法来源、论文来源和代码来源显式对应起来。前端模块详情页展示这些引用，后端以 seed data 或数据库表维护这些引用，后续真实代码接入时按该表落到 plugin registry。

### 2.5.1 模块到论文和代码的对应关系

| 系统模块 | 对应算法/成果 | 本地参考论文 | 代码参考来源 | 展示系统接入位置 |
| --- | --- | --- | --- | --- |
| FedCore | FedAvg、FedDyn、统一 server-client loop | `RFL-NLCP_Robust_Federated_Learning_with_non-IID_Data_and_Limited_Client_Participation (2).pdf`、`FEDCADS_Robust_Federated_Learning_via_Dual_Distillation_and_Participation-Aware_Optimization_Under_non-IID_Data.pdf` | 从各算法代码抽象 `BaseServer`、`BaseClient`、`Trainer`、`StateStore`，并保留 FedAvg/FedDyn 基线实现 | `core/`、`api/overview.py`、`runners/base_runner.py` |
| FedAdapt | FedCVC、FedDTC、RFL-NLCP | `Lai_FedCVC_Federated_Primal-Dual_Learning_with_Client-Driven_Virtual_Compensation_for_Mitigating_CVPRF_2026_paper.pdf`、`基于动态异质性感知与全局梯度偏移校正的联邦对偶优化算法研究-小微格式.docx`、`RFL-NLCP_Robust_Federated_Learning_with_non-IID_Data_and_Limited_Client_Participation (2).pdf` | `plugins/algorithms/fedcvc.py`、`plugins/algorithms/feddtc.py`、`plugins/algorithms/rfl_nlcp.py`，FedCVC/FedDTC 若无完整代码则按论文公式补实现 | `Experiment Builder` 的低参与/异构配置、`DriftPanel` |
| FedOpt | FedCVC、GFed-HSAM、SAM/ESAM/HSAM | `General_Dynamic_Regularization_Federated_Learning_with_Hybrid_Sharpness-Aware_Minimization (1).pdf`、`Lai_FedCVC_Federated_Primal-Dual_Learning_with_Client-Driven_Virtual_Compensation_for_Mitigating_CVPRF_2026_paper.pdf` | `plugins/optimizers/sam.py`、`plugins/optimizers/esam.py`、`plugins/optimizers/hsam.py`、`plugins/algorithms/gfed_hsam.py` | 优化器选择器、算法参数 schema、sharpness 指标展示 |
| FedDistill | FedCADS | `FEDCADS_Robust_Federated_Learning_via_Dual_Distillation_and_Participation-Aware_Optimization_Under_non-IID_Data.pdf` | `plugins/algorithms/fedcads.py`，接入 self/global dual distillation 和 participation-aware dual update | 蒸馏参数表单、表征对齐指标、消融开关 |
| FedGuard | VERT、Krum、Median、Trimmed Mean、FLDetector | `How_to_Defend_Against_Large-Scale_Model_Poisoning_Attacks_in_Federated_Learning_A_Vertical_Solution.pdf`、`软件学报1.pdf` | VERT 按论文实现为 `plugins/defenses/vert.py`；FLBeeline 可参考 `软件学报1—code.zip` 中的 `code1/defenses.py`、`code1/server.py`、`code1/client.py` | `DefensePanel`、攻击模拟配置、可信客户端筛选 |
| FLBeeline | Lightweight large-scale model poisoning defense | `软件学报1.pdf` | `软件学报1—code.zip`，重点文件为 `code1/defenses.py`、`code1/utils/utils.py`、`code1/main-single.py` | FedGuard 前置轻量筛选插件、Top-K 可信梯度可视化 |
| FeaturePoison Attack | Feature-level poisoning、prototype poisoning、FedProto-style prototype attack | FeaturePoison Attack 论文/说明待与本地文件绑定；代码资产以外部仓库为准 | 外部仓库 `https://github.com/bo-lab520/FeaturePoisonAttack`，默认分支 `main`，关键文件包括 `exps/federated_main.py`、`lib/update.py`、`lib/utils.py`、`lib/sampling.py`、`exps/conf.json` | `FeaturePoison Attack` 页面、表征层攻击配置、异常原型检测 |
| FedVision | FedAvg、FedDyn、RFL-NLCP、FedCVC、FedDTC、FedCADS、GFed-HSAM、VERT/FLBeeline | 上述核心算法论文共同支撑 | 复用 `plugins/algorithms/`、`plugins/optimizers/`、`plugins/defenses/`，数据加载放入 `plugins/apps/vision/` | CIFAR/MNIST/AG News 基准页、结果对比页 |
| FedReID | CO-EVO | `2604.26363v1.pdf`，标题为 `CO-EVO: Co-evolving Semantic Anchoring and Style Diversification for Federated DG-ReID` | `plugins/apps/reid/co_evo.py`，若代码未在本目录则先做 ReplayRunner/MockRunner 展示，再补真实实现 | ReID 应用页、mAP/Rank-1 曲线、域泛化结果表 |
| Fed3D | FedULIP | FedULIP 论文/说明待与本地文件绑定；代码资产以外部仓库为准 | 外部仓库 `https://github.com/zjx2024/fedulip_icml`，关键入口包括 `base2new/fedulip_base2new.py`、`cross_dataset/fedulip_cross_dataset.py`、`domain_abcd/fedulip_domain_abcd.py`、`single_dataset/fedulip_single_dataset.py`，并需按 README 将 `fedulip.py` 放入 `ULIP/models/` | 3D 多模态应用页、base-to-new/cross-dataset/domain ABCD 结果 |
| FedUtil | 配置、日志、结果、导出工具 | `FedCompass系统.docx`、各算法论文的实验设置部分 | `configs/`、`scripts/`、`outputs/`，实现实验模板、日志解析、结果导出 | 配置预览、导出按钮、实验复现面板 |

### 2.5.2 代码接入状态字段

每个模块和算法都应维护代码资产状态，方便展示系统区分“可运行”“可回放”“仅论文说明”。

| 状态 | 含义 | 前端展示 |
| --- | --- | --- |
| `paper_only` | 只有论文或方案，暂未接入代码 | 模块详情页显示论文卡片，运行按钮禁用。 |
| `external_code` | 代码在外部仓库，需要克隆或移植 | 显示仓库链接和关键文件列表。 |
| `local_archive` | 代码以本地 zip/压缩包存在 | 显示本地压缩包、关键文件和待解压状态。 |
| `mocked` | 已有演示数据或假数据 runner | 可以运行演示实验，但标注为 mock。 |
| `replay_ready` | 可读取已有日志或结果文件回放 | 支持 Run Monitor 回放和 Results 对比。 |
| `implemented` | 已经接入真实 plugin | 可以创建真实实验。 |
| `validated` | 已通过 smoke test 和指标校验 | 前端显示为可复现实验。 |

### 2.5.3 前端引用展示要求

Modules 页面增加 `ReferencePanel`，每个模块详情必须展示：

| 字段 | 说明 |
| --- | --- |
| `papers` | 本地论文文件名、论文标题、算法名、年份或版本。 |
| `code_sources` | 本地代码包、外部 GitHub 仓库、关键文件路径。 |
| `implementation_status` | `paper_only`、`external_code`、`local_archive`、`mocked`、`replay_ready`、`implemented`、`validated`。 |
| `frontend_views` | 该模块会出现在哪些页面组件里。 |
| `backend_plugins` | 该模块对应的后端 plugin 文件和 hook。 |

### 2.5.4 后端 seed data 建议

后端启动时先用静态 seed data 维护上述映射，后续再迁移到数据库。建议放在：

```text
app/seed/
  modules.json
  algorithms.json
  reference_assets.json
  module_references.json
```

其中 `reference_assets.json` 记录论文、代码仓库、代码压缩包和关键源码文件；`module_references.json` 建立模块与引用资产的多对多关系。

## 3. 前端页面设计

### 3.1 Dashboard 总览页

用途：作为第一屏，展示系统整体状态和能力地图。

主要组件：

| 组件 | 内容 |
| --- | --- |
| SystemStatusBar | 当前实验数量、运行中任务、已注册算法、已注册数据集、可用防御策略。 |
| ModuleCapabilityMap | 以模块矩阵展示 FedCore 到 FedUtil，每个模块显示主算法和能力标签。 |
| RunningExperiments | 展示最近运行的实验，包含状态、算法、数据集、当前轮次、最好精度。 |
| KeyMetricCards | Accuracy、mAP、Rank-1、Attack Detection Accuracy、Communication Cost。 |
| RecentResultsChart | 最近实验的精度/鲁棒性对比柱状图。 |

前端数据来源：

| API | 用途 |
| --- | --- |
| `GET /api/overview` | 获取总览统计。 |
| `GET /api/modules` | 获取模块与算法映射。 |
| `GET /api/experiments?status=running` | 获取运行中实验。 |
| `GET /api/results/recent` | 获取最近结果摘要。 |

### 3.2 Modules 模块能力页

用途：展示每个 FedCompass 模块对应的算法、机制和接口。

页面结构：

| 区域 | 内容 |
| --- | --- |
| 左侧模块列表 | FedCore、FedAdapt、FedOpt、FedDistill、FedGuard、FLBeeline、FeaturePoison Attack、FedVision、FedReID、Fed3D、FedUtil。 |
| 中部模块详情 | 模块定位、解决问题、主要算法、可选基线、输入输出。 |
| 右侧实现状态 | `planned`、`mocked`、`implemented`、`validated`。 |
| 论文代码追溯 | `ReferencePanel` 展示本地论文、外部仓库、本地代码包、关键源码文件和接入状态。 |
| 底部接口示意 | 该模块对应 backend service、plugin hook、配置字段。 |

模块到前端展示内容：

| 模块 | 页面展示重点 |
| --- | --- |
| FedCore | Server、Client、Trainer、StateStore、Registry 的训练流程图。 |
| FedAdapt | Non-IID、低参与率、silence table、EMA 异质性、梯度漂移补偿。 |
| FedOpt | FedCVC、GFed-HSAM、FedDyn、FedProx、SCAFFOLD、SAM/ESAM/HSAM 对比。 |
| FedDistill | self/global dual distillation、动态蒸馏权重、特征对齐。 |
| FedGuard | VERT、Krum、Median、Trimmed Mean、FLDetector 和攻击模拟。 |
| FLBeeline | 历史梯度几何一致性、可信客户端筛选、低开销防御。 |
| FeaturePoison Attack | 特征层投毒、原型异常检测、表征安全。 |
| FedVision | CIFAR/MNIST/AG News、IID/Dirichlet/Pathological split。 |
| FedReID | CO-EVO、semantic anchor、global style bank、mAP/Rank-1。 |
| Fed3D | FedULIP、ULIP backbone、adapter/SACA、跨数据集评测。 |
| FedUtil | 配置、日志、checkpoint、结果表格、批量任务。 |

### 3.3 Experiment Builder 实验配置页

用途：让用户通过表单组合一个可运行演示实验。

配置区块：

| 区块 | 字段 |
| --- | --- |
| Task | `vision_classification`、`reid`、`ulip3d`、`defense_demo`。 |
| Dataset | CIFAR-10、MNIST、AG News、Market1501、DukeMTMC、ModelNet40 等。 |
| Split | IID、Dirichlet alpha、Pathological、domain-as-client。 |
| Algorithm | FedAvg、FedDyn、RFL-NLCP、FedCVC、FedDTC、FedCADS、GFed-HSAM。 |
| Optimizer | SGD、SAM、ESAM、HSAM。 |
| Defense | None、VERT、Krum、Median、Trimmed Mean、FLDetector、FLBeeline。 |
| Attack | None、GN、MR、AGR、ALIE、FeaturePoison。 |
| Participation | 客户端数量、每轮参与率、沉默客户端比例。 |
| Training | rounds、local epochs、batch size、learning rate、seed。 |

交互设计：

| 行为 | 说明 |
| --- | --- |
| 选择模块后过滤算法 | 例如选择 FedDistill 时优先展示 FedCADS 和蒸馏相关参数。 |
| 参数联动 | 选择 VERT 后显示攻击配置和可信筛选阈值。 |
| 配置预览 | 实时生成 YAML/JSON 配置预览。 |
| 一键运行 | 调用后端创建 experiment job。 |
| 一键加载模板 | 提供低参与率、投毒防御、ReID、FedULIP 四类预设。 |

### 3.4 Run Monitor 实验监控页

用途：展示实验运行过程。

主要组件：

| 组件 | 内容 |
| --- | --- |
| RoundProgress | 当前轮次、总轮次、预计剩余时间。 |
| MetricCurves | train loss、test accuracy、mAP、Rank-1、detection accuracy。 |
| ClientParticipationHeatmap | 每轮客户端参与情况、沉默状态、恶意客户端标记。 |
| DriftPanel | FedDTC/FedCVC 的异质性指标、梯度漂移、补偿量。 |
| DefensePanel | VERT/FLBeeline 的可信客户端集合、被过滤客户端、攻击检测指标。 |
| LogConsole | 后端训练日志流。 |

实时数据来源：

| API | 用途 |
| --- | --- |
| `GET /api/experiments/{id}` | 实验元数据。 |
| `GET /api/experiments/{id}/metrics` | 历史指标。 |
| `GET /api/experiments/{id}/events/stream` | 实时事件流。 |
| `POST /api/experiments/{id}/stop` | 停止实验。 |

### 3.5 Results 结果对比页

用途：沉淀实验结果，支持论文式对比展示。

核心能力：

| 能力 | 说明 |
| --- | --- |
| 实验筛选 | 按任务、数据集、算法、攻击、防御、参与率、seed 过滤。 |
| 曲线对比 | 多算法 accuracy/loss/robustness 曲线叠加。 |
| 表格对比 | 输出 best accuracy、final accuracy、mAP、Rank-1、detection accuracy、communication rounds。 |
| 消融视图 | 对比是否启用 compensation、distillation、HSAM、defense。 |
| 导出 | 导出 CSV、JSON、Markdown 表格和 PNG 图。 |

### 3.6 Apps 应用场景页

用途：展示 FedVision、FedReID、Fed3D 三个应用方向。

| 应用 | 页面内容 |
| --- | --- |
| FedVision | 分类任务样例、异构划分可视化、基础算法对比。 |
| FedReID | ReID 数据域、CO-EVO 机制、mAP/Rank-1 对比图。 |
| Fed3D | 点云/文本/图像多模态示意、FedULIP 评测协议、base-to-new/cross-dataset 结果。 |

## 4. 后端模块设计

推荐目录结构：

```text
fedcompass_demo/
  app/
    main.py
    api/
      overview.py
      modules.py
      algorithms.py
      references.py
      datasets.py
      experiments.py
      results.py
      events.py
    core/
      config.py
      database.py
      event_bus.py
      registry.py
    models/
      module.py
      algorithm.py
      reference.py
      dataset.py
      experiment.py
      metric.py
      artifact.py
    schemas/
      module.py
      algorithm.py
      reference.py
      experiment.py
      result.py
    services/
      module_service.py
      reference_service.py
      experiment_service.py
      result_service.py
      artifact_service.py
    runners/
      base_runner.py
      mock_runner.py
      fedcompass_runner.py
    plugins/
      algorithms/
      optimizers/
      defenses/
      apps/
    workers/
      task_queue.py
      experiment_worker.py
  outputs/
  configs/
  migrations/
```

### 4.1 API 层

| 模块 | 职责 |
| --- | --- |
| `overview.py` | 汇总系统状态、最近实验、关键指标。 |
| `modules.py` | 返回模块、算法映射和模块实现状态。 |
| `algorithms.py` | 返回算法注册表、参数 schema、适用任务。 |
| `references.py` | 返回论文、代码仓库、本地代码包和模块引用关系。 |
| `datasets.py` | 返回数据集、划分策略、任务类型。 |
| `experiments.py` | 创建、查询、停止、复制实验。 |
| `results.py` | 查询指标、对比结果、导出表格。 |
| `events.py` | 提供 SSE/WebSocket 实时事件流。 |

### 4.2 Service 层

| Service | 职责 |
| --- | --- |
| `ModuleService` | 管理 FedCompass 模块信息和算法映射。 |
| `ReferenceService` | 管理论文与代码资产，提供模块、算法、页面组件的引用追溯数据。 |
| `ExperimentService` | 校验配置、创建任务、更新状态。 |
| `ResultService` | 聚合指标、生成对比数据、输出导出文件。 |
| `ArtifactService` | 管理日志、checkpoint、图表、配置文件。 |
| `RegistryService` | 注册算法、优化器、防御插件、应用插件。 |

### 4.3 Runner 层

| Runner | 用途 |
| --- | --- |
| `MockRunner` | 前期展示系统先跑假数据，快速完成可演示闭环。 |
| `FedCompassRunner` | 对接真实 `fedcompass/` 训练代码，读取 YAML 配置并启动实验。 |
| `ReplayRunner` | 读取已有日志和 result.json，回放训练过程，适合答辩和展示。 |

建议第一版优先实现 `MockRunner + ReplayRunner`，这样前端能很快成型；真实训练接入放第二阶段。

## 5. 核心数据模型

### 5.1 Module

```json
{
  "id": "fedadapt",
  "name": "FedAdapt",
  "positioning": "动态异质性感知与低参与补偿",
  "capabilities": ["heterogeneity", "low_participation", "drift_compensation"],
  "primary_algorithms": ["FedCVC", "FedDTC", "RFL-NLCP"],
  "baseline_algorithms": ["FedAvg", "FedProx"],
  "status": "implemented"
}
```

### 5.2 Algorithm

```json
{
  "id": "fedcvc",
  "name": "FedCVC",
  "module_ids": ["fedadapt", "fedopt"],
  "task_types": ["vision_classification"],
  "parameters": {
    "dual_lr": {"type": "number", "default": 0.01},
    "compensation_weight": {"type": "number", "default": 1.0},
    "silence_window": {"type": "integer", "default": 5}
  },
  "hooks": ["before_aggregate", "after_local_train", "update_global_state"]
}
```

### 5.3 Experiment

```json
{
  "id": "exp_20260623_001",
  "name": "FedCVC on CIFAR10 Dirichlet alpha=0.1",
  "task_type": "vision_classification",
  "dataset": "cifar10",
  "algorithm": "fedcvc",
  "optimizer": "sam",
  "defense": "none",
  "attack": "none",
  "status": "running",
  "current_round": 38,
  "total_rounds": 100,
  "config": {},
  "created_at": "2026-06-23T16:00:00"
}
```

### 5.4 MetricEvent

```json
{
  "experiment_id": "exp_20260623_001",
  "round": 38,
  "metrics": {
    "train_loss": 0.812,
    "test_accuracy": 0.746,
    "communication_cost_mb": 142.3,
    "client_participation_rate": 0.1,
    "gradient_drift": 0.037
  }
}
```

### 5.5 ReferenceAsset

```json
{
  "id": "paper_fedcvc_2026",
  "asset_type": "paper",
  "title": "FedCVC: Federated Primal-Dual Learning with Client-Driven Virtual Compensation for Mitigating Dual Drift",
  "local_path": "Lai_FedCVC_Federated_Primal-Dual_Learning_with_Client-Driven_Virtual_Compensation_for_Mitigating_CVPRF_2026_paper.pdf",
  "external_url": null,
  "related_modules": ["fedadapt", "fedopt"],
  "related_algorithms": ["fedcvc"],
  "status": "paper_only"
}
```

代码资产示例：

```json
{
  "id": "repo_feature_poison_attack",
  "asset_type": "external_code",
  "title": "bo-lab520/FeaturePoisonAttack",
  "local_path": null,
  "external_url": "https://github.com/bo-lab520/FeaturePoisonAttack",
  "default_branch": "main",
  "key_files": [
    "exps/federated_main.py",
    "lib/update.py",
    "lib/utils.py",
    "lib/sampling.py",
    "exps/conf.json"
  ],
  "related_modules": ["featurepoison_attack"],
  "related_algorithms": ["feature_poison", "fedproto_prototype_attack"],
  "status": "external_code"
}
```

```json
{
  "id": "repo_fedulip_icml",
  "asset_type": "external_code",
  "title": "zjx2024/fedulip_icml",
  "local_path": null,
  "external_url": "https://github.com/zjx2024/fedulip_icml",
  "default_branch": "main",
  "key_files": [
    "base2new/fedulip_base2new.py",
    "cross_dataset/fedulip_cross_dataset.py",
    "domain_abcd/fedulip_domain_abcd.py",
    "single_dataset/fedulip_single_dataset.py",
    "ULIP/models/fedulip.py"
  ],
  "related_modules": ["fed3d"],
  "related_algorithms": ["fedulip"],
  "status": "external_code"
}
```

## 6. API 草案

### 6.1 Overview

| Method | Path | 说明 |
| --- | --- | --- |
| GET | `/api/overview` | 系统总览统计。 |
| GET | `/api/overview/capability-map` | 模块能力图数据。 |

### 6.2 Modules and Algorithms

| Method | Path | 说明 |
| --- | --- | --- |
| GET | `/api/modules` | 模块列表。 |
| GET | `/api/modules/{module_id}` | 模块详情。 |
| GET | `/api/algorithms` | 算法列表。 |
| GET | `/api/algorithms/{algorithm_id}` | 算法详情和参数 schema。 |
| GET | `/api/optimizers` | 优化器列表。 |
| GET | `/api/defenses` | 防御算法列表。 |
| GET | `/api/attacks` | 攻击模拟列表。 |
| GET | `/api/references` | 全部论文与代码资产。 |
| GET | `/api/modules/{module_id}/references` | 某个模块对应的论文与代码资产。 |
| GET | `/api/algorithms/{algorithm_id}/references` | 某个算法对应的论文与代码资产。 |

### 6.3 Experiments

| Method | Path | 说明 |
| --- | --- | --- |
| POST | `/api/experiments` | 创建实验。 |
| GET | `/api/experiments` | 实验列表。 |
| GET | `/api/experiments/{id}` | 实验详情。 |
| POST | `/api/experiments/{id}/start` | 启动实验。 |
| POST | `/api/experiments/{id}/stop` | 停止实验。 |
| POST | `/api/experiments/{id}/clone` | 复制实验配置。 |
| GET | `/api/experiments/{id}/metrics` | 获取指标曲线。 |
| GET | `/api/experiments/{id}/logs` | 获取训练日志。 |
| GET | `/api/experiments/{id}/events/stream` | SSE 实时事件流。 |

### 6.4 Results

| Method | Path | 说明 |
| --- | --- | --- |
| GET | `/api/results` | 结果列表。 |
| POST | `/api/results/compare` | 多实验对比。 |
| GET | `/api/results/{id}/summary` | 单实验结果摘要。 |
| GET | `/api/results/export` | 导出 CSV/JSON/Markdown。 |

## 7. 数据库表设计

| 表 | 主要字段 | 说明 |
| --- | --- | --- |
| `modules` | id、name、positioning、description、status | FedCompass 模块。 |
| `algorithms` | id、name、category、description、parameter_schema | 算法注册表。 |
| `module_algorithms` | module_id、algorithm_id、role | 模块和算法多对多关系，role 可为 primary/baseline/auxiliary。 |
| `reference_assets` | id、asset_type、title、local_path、external_url、metadata_json、status | 论文、外部仓库、本地代码包和关键源码文件。 |
| `module_references` | module_id、reference_asset_id、role | 模块与引用资产的多对多关系，role 可为 paper/code/baseline/evidence。 |
| `algorithm_references` | algorithm_id、reference_asset_id、role | 算法与引用资产的多对多关系。 |
| `datasets` | id、name、task_type、split_options | 数据集和划分策略。 |
| `experiments` | id、name、task_type、status、config_json、created_at、finished_at | 实验主表。 |
| `experiment_metrics` | id、experiment_id、round、metric_name、metric_value | 指标曲线。 |
| `experiment_events` | id、experiment_id、event_type、payload_json、created_at | 实时事件和日志。 |
| `artifacts` | id、experiment_id、artifact_type、path、metadata_json | 配置、日志、checkpoint、导出结果。 |

## 8. 前后端字段对齐

| 展示需求 | 前端组件 | 后端数据 |
| --- | --- | --- |
| 模块能力矩阵 | `ModuleCapabilityMap` | `GET /api/modules` |
| 算法参数表单 | `AlgorithmConfigForm` | `GET /api/algorithms/{id}` 的 `parameter_schema` |
| 论文代码追溯 | `ReferencePanel` | `GET /api/modules/{module_id}/references`、`reference_assets` |
| 实验状态 | `ExperimentStatusBadge` | `experiments.status` |
| 训练曲线 | `MetricCurves` | `experiment_metrics` |
| 客户端参与热力图 | `ClientParticipationHeatmap` | `experiment_events` 中的 `client_participation` |
| 攻击检测面板 | `DefensePanel` | `experiment_metrics` 和 `experiment_events` |
| 结果对比表 | `ResultComparisonTable` | `POST /api/results/compare` |

## 9. 演示系统实现阶段

### 第 1 阶段：静态能力展示

实现内容：

| 前端 | 后端 |
| --- | --- |
| Dashboard、Modules、Apps 三个页面 | FastAPI 基础工程、模块/算法/引用资产静态 JSON API |
| 模块能力矩阵 | `GET /api/modules`、`GET /api/algorithms` |
| 算法详情抽屉、ReferencePanel | 内置模块算法映射和论文代码资产 seed data |

交付效果：可以完整展示 FedCompass 系统架构和模块算法匹配。

### 第 2 阶段：实验配置与假数据运行

实现内容：

| 前端 | 后端 |
| --- | --- |
| Experiment Builder | `POST /api/experiments` |
| Run Monitor | `MockRunner` 生成 round-level metrics |
| 训练曲线、日志流、参与热力图 | SSE 推送 metric/log event |

交付效果：可以模拟启动实验，并动态展示训练过程。

### 第 3 阶段：真实日志回放

实现内容：

| 前端 | 后端 |
| --- | --- |
| Results 对比页 | `ReplayRunner` 读取已有 result.json/log |
| 多实验筛选和对比 | 指标聚合与导出 |
| 防御实验展示 | VERT/FLBeeline/FeaturePoison Attack 攻击检测结果回放 |

交付效果：可以把已有论文实验结果接入展示系统。

### 第 4 阶段：真实训练接入

实现内容：

| 前端 | 后端 |
| --- | --- |
| 完整实验生命周期管理 | `FedCompassRunner` 调用真实训练脚本 |
| 停止/复制/导出实验 | 后台任务队列、artifact 管理 |
| Vision/ReID/3D 应用演示 | 对接 apps/vision、apps/reid、apps/ulip3d |

交付效果：展示系统变成可运行的轻量实验平台。

## 10. 最小可行版本清单

MVP 建议先做这些：

| 类型 | 必做项 |
| --- | --- |
| 前端页面 | Dashboard、Modules、Experiment Builder、Run Monitor、Results。 |
| 后端 API | modules、algorithms、references、experiments、metrics、events。 |
| Runner | MockRunner。 |
| 数据 | 模块算法映射、论文代码资产映射、3 个预设实验、2 组模拟指标。 |
| 图表 | 精度曲线、loss 曲线、算法对比柱状图、客户端参与热力图。 |
| 导出 | Markdown/CSV 结果表。 |

MVP 完成后，系统就可以用于项目展示、开题/答辩演示和后续真实算法接入。
