# FedCompass 前后端接口文档

> 面向前端开发人员。Base URL: `http://localhost:8000`

---

## 一、系统整体运行流程

```
前端 (React + Vite)                          后端 (FastAPI :8000)
═══════════════════                          ════════════════════

1. Dashboard 加载
   GET /api/overview ──────────────────→ 查询数据库统计 + 最近 5 个实验
   GET /api/overview/capability-map ──→ 返回 11 个模块简要信息
   每 30s 重新 GET /api/overview 刷新状态

2. Modules 页
   GET /api/modules ──────────────────→ 返回 11 个模块 + 嵌套的关联算法
   用户点击某模块 → 从已有数据渲染详情（无需额外请求）
   切换到 References 标签 → GET /api/modules/{id}/references

3. Experiment Builder
   页面加载时并行请求 5 个下拉数据源:
     GET /api/algorithms ────────────→ 算法列表
     GET /api/datasets ─────────────→ 数据集列表
     GET /api/optimizers ───────────→ 优化器列表
     GET /api/defenses ─────────────→ 防御策略列表
     GET /api/attacks ──────────────→ 攻击模拟列表
   「一键加载模板」按钮:
     GET /api/experiments/templates → 4 个预设配置
   用户点击「运行」:
     POST /api/experiments ────────→ 创建实验 (status=draft) → 返回 ExperimentOut
     POST /api/experiments/{id}/start → 后台启动 MockRunner → 返回 {"status":"started"}
     跳转到 Run Monitor 页

4. Run Monitor 实时监控
   GET /api/experiments/{id} ───────→ 实验元数据
   GET /api/experiments/{id}/metrics → 已有指标曲线（历史数据）
   建立 SSE 连接:
     EventSource('/api/experiments/{id}/events/stream')
       收到事件:
         round_complete      → 更新轮次进度 + 追加曲线数据点
         client_participation → 更新热力图
         defense_detection   → 更新防御面板
         completed           → 显示最终指标
         stopped             → 显示停止信息
         done                → 关闭 EventSource

5. Results 对比
   GET /api/experiments ─────────────→ 实验列表（左侧筛选）
   GET /api/results?algorithm=fedcvc → 按条件过滤结果
   POST /api/results/compare ───────→ 选中实验对比
   GET /api/results/export?format=csv → 导出文件下载
```

### 关键设计决策

| 决策 | 说明 |
| --- | --- |
| 异步实验执行 | `POST /start` 立即返回，实验在后台线程运行，前端通过 SSE 订阅进度 |
| 指标双通道 | SSE 推送实时事件（轻量、每轮一次），`GET /metrics` 获取完整历史曲线（重量、用于初始加载和页面刷新恢复） |
| 状态机 | draft → running → completed / failed / stopped |
| 数据库 | SQLite，启动时自动建表 + seed 全部静态数据 |
| CORS | 已配置允许 `localhost:5173` 和 `localhost:3000` |

---

## 二、页面与 API 详细映射

### 2.1 Dashboard 总览页

| 页面组件 | 调用 API | 刷新策略 |
| --- | --- | --- |
| SystemStatusBar | `GET /api/overview` | 页面加载 + 每 30s 轮询 |
| ModuleCapabilityMap | `GET /api/overview/capability-map` | 页面加载（静态数据） |
| RunningExperiments | `GET /api/experiments?status=running` | 页面加载 + 每 10s 轮询 |
| KeyMetricCards | 来自 `GET /api/overview` 的返回值 | 同 SystemStatusBar |
| RecentResultsChart | `GET /api/results/recent?limit=10` | 页面加载 |

**前端实现要点：**
- `overview` 和 `experiments` 可并行请求（Promise.all）
- `capability-map` 返回模块数组，每个含 `id` / `name` / `positioning` / `status`，渲染模块矩阵

---

### 2.2 Modules 模块能力页

| 页面区域 | 调用 API | 说明 |
| --- | --- | --- |
| 左侧模块列表 | `GET /api/modules` | 返回完整数据（含嵌套 algorithms），前端直接渲染列表 |
| 中部模块详情 | 同上 | 从列表中取当前选中模块，无需额外请求 |
| 右侧实现状态 | 同上 | 使用 `status` 字段渲染状态标签 |
| ReferencePanel | `GET /api/modules/{id}/references` | 用户切换到 References 标签时才请求（懒加载） |

**重要：** `/api/modules` 已携带 `algorithms` 嵌套数组，每个算法含 `id` / `name` / `category` / `status`。`/api/modules/{id}` 仅用于单独刷新某个模块的场景。

---

### 2.3 Experiment Builder 实验配置页

**页面加载时并行请求（5 个 API 同时发出）：**

| API | 用途 | 缓存策略 |
| --- | --- | --- |
| `GET /api/algorithms` | 算法下拉选项 | staleTime: 5min |
| `GET /api/datasets` | 数据集下拉选项 | staleTime: 5min |
| `GET /api/optimizers` | 优化器下拉选项（返回 category=optimization 的算法） | staleTime: 5min |
| `GET /api/defenses` | 防御策略下拉选项（返回 category=defense 的算法） | staleTime: 5min |
| `GET /api/attacks` | 攻击模拟下拉选项（返回 category=attack 的算法） | staleTime: 5min |
| `GET /api/experiments/templates` | 4 个预设模板 | staleTime: 5min |

**一键加载模板流程：**
1. 调用 `GET /api/experiments/templates` 获取模板列表
2. 用户选择一个模板 → 将其 `config` 对象填充到表单各字段
3. 用户可在此基础上修改，然后点击运行

**用户点击运行后流程：**
```
1. POST /api/experiments (body: ExperimentConfig)
   → Response: ExperimentOut (status="draft")

2. POST /api/experiments/{id}/start
   → Response: {"status": "started"}

3. 前端跳转到 /monitor/{id}
```

**ExperimentConfig 请求体完整参数**

| 字段 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| task_type | string | 否 | `"vision_classification"` | vision_classification \| reid \| ulip3d \| defense_demo |
| dataset | string | 否 | `"cifar10"` | 数据集 ID，从 `/api/datasets` 获得 |
| split | string | 否 | `"iid"` | iid \| dirichlet \| pathological \| domain_as_client |
| algorithm | string | 否 | `"fedavg"` | 算法 ID，从 `/api/algorithms` 获得 |
| optimizer | string | 否 | `"sgd"` | 优化器 ID，从 `/api/optimizers` 获得 |
| defense | string | 否 | `"none"` | 防御策略 ID，从 `/api/defenses` 获得 |
| attack | string | 否 | `"none"` | 攻击模拟 ID，从 `/api/attacks` 获得 |
| num_clients | int | 否 | 100 | 客户端总数 |
| participation_rate | float | 否 | 0.1 | 每轮参与比例 (0~1) |
| silence_rate | float | 否 | 0.0 | 静默客户端比例 (0~1) |
| rounds | int | 否 | 100 | 总训练轮数 |
| local_epochs | int | 否 | 5 | 每轮本地训练 epoch |
| batch_size | int | 否 | 64 | batch size |
| learning_rate | float | 否 | 0.01 | 学习率 |
| seed | int | 否 | 42 | 随机种子 |

**预设模板**

| 模板 ID | 名称 | 关键配置 |
| --- | --- | --- |
| low_participation | 低参与率非IID场景 | algorithm=fedcvc, split=dirichlet, participation=0.1, silence=0.3 |
| poison_defense | 投毒攻击与防御对比 | task_type=defense_demo, defense=vert, attack=feature_poison |
| reid_baseline | ReID 域泛化基准 | task_type=reid, dataset=market1501, algorithm=co_evo |
| fedulip_3d | FedULIP 3D 多模态 | task_type=ulip3d, dataset=modelnet40, algorithm=fedulip |

---

### 2.4 Run Monitor 实验监控页

**核心交互逻辑：**

```
// 页面加载
const exp  = await fetch('/api/experiments/{id}')     // 实验元数据
const data = await fetch('/api/experiments/{id}/metrics') // 历史指标曲线

// 建立 SSE 连接
const es = new EventSource('/api/experiments/{id}/events/stream')

es.addEventListener('round_complete', (e) => {
  const { round, train_loss, test_accuracy, timestamp } = JSON.parse(e.data)
  // 更新 RoundProgress 组件: current_round = round
  // 追加 MetricCurves 数据点
})

es.addEventListener('client_participation', (e) => {
  const { round, participating, silent, malicious } = JSON.parse(e.data)
  // 更新 ClientParticipationHeatmap
})

es.addEventListener('defense_detection', (e) => {
  const { round, trusted_clients, filtered_clients, detection_accuracy } = JSON.parse(e.data)
  // 更新 DefensePanel
})

es.addEventListener('completed', (e) => {
  const { final_accuracy, final_loss, message } = JSON.parse(e.data)
  // 显示最终指标，标记实验完成
})

es.addEventListener('done', () => {
  es.close() // 关闭连接
})
```

**SSE 事件类型与 Payload 详细说明**

| event | 触发频率 | Payload 字段 |
| --- | --- | --- |
| round_complete | 每轮（约 0.3s/轮） | `round` int — 当前轮次, `train_loss` float, `test_accuracy` float, `timestamp` string (ISO 8601) |
| client_participation | 每 5 轮 | `round` int, `participating` int[] — 参与客户端 ID 列表, `silent` int[] — 静默客户端, `malicious` int[] — 恶意客户端（第 3 轮后才有） |
| defense_detection | 每 10 轮（第 6 轮起） | `round` int, `trusted_clients` int[] — 可信客户端, `filtered_clients` int[] — 被过滤客户端, `detection_accuracy` float |
| completed | 实验正常结束 | `final_accuracy` float, `final_loss` float, `message` string |
| stopped | 用户手动停止 | `round` int — 停止时的轮次, `message` string |
| done | SSE 流结束 | `status` string — "completed" 或 "stopped" |

**前端实现要点：**
- SSE 事件数据用于实时追加，`GET /metrics` 用于初始加载曲线（避免丢失 SSE 连接前的历史数据点）
- 如果页面刷新，重新调用 `GET /metrics` 恢复所有数据点，再重新建立 SSE
- 点击 Stop → `POST /api/experiments/{id}/stop` → 后端设 `_stop=True`，Runner 在下一轮退出并推送 stopped 事件
- SSE 连接断开时自动重连（EventSource 默认行为）

---

### 2.5 Results 结果对比页

| 操作 | 调用 API | 说明 |
| --- | --- | --- |
| 筛选条件加载 | `GET /api/algorithms` + `GET /api/datasets` | 填充筛选下拉框 |
| 实验结果列表 | `GET /api/results?algorithm=fedcvc&dataset=cifar10&limit=50` | 支持 task_type / algorithm / dataset 三参数过滤 |
| 选中对比 | `POST /api/results/compare` | 传入 `{"experiment_ids": [...]}`，返回对比数组 |
| 最近结果 | `GET /api/results/recent?limit=10` | Dashboard 和首页快速加载 |
| 导出 CSV | `GET /api/results/export?format=csv&experiment_ids=exp_001,exp_002` | 浏览器触发文件下载 |
| 导出 Markdown | `GET /api/results/export?format=markdown` | 不传 experiment_ids 则导出全部已完成实验 |
| 导出 JSON | `GET /api/results/export?format=json` | 同上 |

**CompareResult 返回结构**

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| experiment_id | string | 实验 ID |
| experiment_name | string | 实验名称 |
| algorithm | string \| null | 算法 ID |
| dataset | string \| null | 数据集 ID |
| best_accuracy | float \| null | test_accuracy 全轮次最大值 |
| final_accuracy | float \| null | test_accuracy 最后一轮值 |
| best_loss | float \| null | train_loss 全轮次最小值 |
| communication_rounds | int | 当前/已完成轮次 |
| status | string | completed / running / stopped / failed |

**注意：** `best_accuracy` ≠ `final_accuracy`。前者是训练过程中的峰值，后者是最终收敛值。两列在对比表中都应展示。

---

### 2.6 Apps 应用场景页

| 应用 | 数据来源 | 说明 |
| --- | --- | --- |
| FedVision | `GET /api/modules/fedvision` + `GET /api/results?task_type=vision_classification` | 展示分类任务下已完成实验的对比图表 |
| FedReID | `GET /api/modules/fedreid` + `GET /api/results?task_type=reid` | 展示 ReID 任务的 mAP/Rank-1 对比 |
| Fed3D | `GET /api/modules/fed3d` + `GET /api/results?task_type=ulip3d` | 展示 3D 多模态实验结果 |

如果对应 task_type 还没有实验数据，前端展示模块基础信息，图表区域显示空状态并引导去 Experiment Builder 创建。

---

## 三、核心数据模型速查

### Experiment 状态机

```
draft ──POST /start──→ running ──完成──→ completed
                           │
                           ├──POST /stop──→ stopped
                           └──异常──────→ failed (MockRunner 不触发)
```

### Module / Algorithm.status 枚举

| 值 | 含义 | 前端展示 |
| --- | --- | --- |
| `paper_only` | 只有论文 | 灰色标签，运行按钮 disabled |
| `external_code` | 代码在 GitHub | 显示外部链接 |
| `local_archive` | 本地有 zip | 显示本地路径 |
| `mocked` | MockRunner 可运行 | 绿色标签，可模拟运行 |
| `replay_ready` | 可回放结果 | 蓝色标签 |
| `implemented` | 真实代码接入 | 绿色标签 |
| `validated` | 已验证 | 绿色标签 + ✓ |

### Algorithm.category 与前端下拉对应

| category 值 | 对应前端下拉 | 示例算法 |
| --- | --- | --- |
| aggregation | 算法下拉 | FedAvg, FedDyn |
| compensation | 算法下拉 | FedCVC, FedDTC, RFL-NLCP |
| regularization | 算法下拉 | FedProx, SCAFFOLD |
| optimization | 优化器下拉 → `GET /api/optimizers` | GFed-HSAM |
| distillation | 算法下拉 | FedCADS |
| defense | 防御下拉 → `GET /api/defenses` | VERT, Krum, Median, FLBeeline... |
| attack | 攻击下拉 → `GET /api/attacks` | FeaturePoison, FedProto Prototype Attack |
| reid | 算法下拉 | CO-EVO |
| 3d_multimodal | 算法下拉 | FedULIP |

---

## 四、指标数据说明

### GET /api/experiments/{id}/metrics 返回的 5 条曲线

| metric_name | 含义 | 值范围 | 前端使用 |
| --- | --- | --- | --- |
| train_loss | 训练损失 | 通常 2.0 → 0.05 | MetricCurves 的 loss 曲线 |
| test_accuracy | 测试准确率 | 0.1 → 0.95 | MetricCurves 的 accuracy 曲线，也是 Results 对比的核心指标 |
| communication_cost_mb | 每轮通信开销 | ~140 MB | KeyMetricCards 通信量指标 |
| client_participation_rate | 客户端参与率 | 0.05~0.2（取决于配置） | ClientParticipationHeatmap 的背景数据 |
| gradient_drift | 梯度漂移量 | 0.05 → 0.001 | DriftPanel 的漂移曲线 |

---

## 五、错误处理

| 状态码 | 含义 | 前端处理建议 |
| --- | --- | --- |
| 200 | 成功 | - |
| 400 | 参数错误（如对非 draft 实验调 start） | toast 提示 `detail` 内容 |
| 404 | 资源不存在 | 跳转到对应空状态页或显示"未找到" |
| 422 | 请求体校验失败 | 表单字段标红，显示校验错误 |

错误响应统一格式：`{"detail": "错误描述字符串"}`

---

## 六、前端状态管理建议

### TanStack Query 配置

| queryKey | API | staleTime | 说明 |
| --- | --- | --- | --- |
| `['overview']` | `GET /api/overview` | 30s | 动态统计 |
| `['modules']` | `GET /api/modules` | 5min | 静态数据 |
| `['algorithms']` | `GET /api/algorithms` | 5min | 静态数据 |
| `['datasets']` | `GET /api/datasets` | 5min | 静态数据 |
| `['experiments']` | `GET /api/experiments` | 10s | 动态列表 |
| `['experiment', id]` | `GET /api/experiments/{id}` | 5s | 单个实验 |
| `['metrics', id]` | `GET /api/experiments/{id}/metrics` | 5s | 指标曲线 |
| `['results']` | `GET /api/results` | 10s | 结果列表 |
| `['templates']` | `GET /api/experiments/templates` | 5min | 预设模板 |

### Zustand 局部状态

| 状态 | 类型 | 用途 |
| --- | --- | --- |
| currentExperimentId | string \| null | Run Monitor 当前监控的实验 |
| selectedCompareIds | string[] | Results 页已选中的对比实验 |
| sseConnected | boolean | SSE 连接状态 |

---

## 七、Vite 代理配置

```ts
// vite.config.ts
export default defineConfig({
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      }
    }
  }
})
```

后端 CORS 已配置允许 `localhost:5173` 和 `localhost:3000`，代理和直连均可。

---

## 八、快速启动

```bash
# 后端
cd fedcompass_backend
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Swagger: http://localhost:8000/docs
```

启动后自动建表 + seed 全部静态数据（11 模块、19 算法、12 引用资产、7 数据集、4 个预设模板）。