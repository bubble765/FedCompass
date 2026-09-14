# FedCompass 前端原型

这是一个不依赖打包工具的静态前端原型，直接根据 `FedCompass展示系统设计.md` 和 `0623李-F&B_API_Doc.md` 落了一版页面结构与接口映射。当前运行监控页已经接入四智能体协同工作流：实验场景智能体、运行保障智能体、安全监控智能体、结果分析智能体。

## 页面

- `Dashboard`
- `Modules`
- `Experiment Builder`
- `Run Monitor`
- `Results`
- `Apps`

## 接口映射

页面里已经接好了这些接口名，后端可用时会直接请求，失败时自动回退到本地 mock 数据：

- `GET /api/overview`
- `GET /api/overview/capability-map`
- `GET /api/modules`
- `GET /api/modules/{id}`
- `GET /api/modules/{id}/references`
- `GET /api/algorithms`
- `GET /api/datasets`
- `GET /api/optimizers`
- `GET /api/defenses`
- `GET /api/attacks`
- `GET /api/experiments/templates`
- `POST /api/experiments`
- `POST /api/experiments/{id}/preflight`
- `GET /api/experiments/{id}/workflow`
- `GET /api/experiments/{id}/operations-report`
- `GET /api/experiments/{id}/security-report`
- `GET /api/experiments/{id}/result-analysis`
- `POST /api/experiments/{id}/result-analysis/regenerate`
- `POST /api/experiments/{id}/start`
- `GET /api/experiments`
- `GET /api/experiments/{id}`
- `GET /api/experiments/{id}/metrics`
- `GET /api/experiments/{id}/events/stream`
- `POST /api/experiments/{id}/stop`
- `GET /api/results`
- `GET /api/results/recent`
- `POST /api/results/compare`
- `GET /api/results/export`

## 使用方式

当前联调流程为：配置页创建实验并调用运行预检；运行监控页展示四个智能体的阶段、状态、工具调用次数和证据卡片；用户点击“确认并启动”后，Runner 在运行中的每个检查周期调用运行保障智能体和安全监控智能体，并通过 SSE 推送轮次、安全观测、周期检查和智能体更新；实验结束后显示结果分析智能体生成的观察、可支持结论、不能推出的结论、限制和补充实验建议。

如果本机有 Python，可以在当前目录启动一个静态服务：

```bash
cd /Users/bubble/Desktop/FedCompass_副本/0624v1冠-前端/FL系统/frontend
/Users/bubble/Desktop/FedCompass_副本/.venv-fedcompass/bin/python -m http.server 4173
```

然后打开 `http://localhost:4173`。

默认是 Mock 模式，可以在左侧切换为 API 优先模式，直接对接 `http://localhost:8000`。
