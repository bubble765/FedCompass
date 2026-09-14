# FedCompass Docker 部署构建计划

> 文档状态：前后端与本地 Qwen3-0.6B 模型容器配置已落实，镜像构建待 Docker 环境验证
>
> 编制日期：2026-07-19
>
> 目标：将 FedCompass 前后端、配置助手、实验运行能力和必要数据整理为可迁移到其他服务器的 Docker 部署方案。

## 1. 目标与范围

### 1.1 目标

- 在干净的 Linux 服务器上通过 Docker Compose 启动完整系统和本地模型服务。
- 浏览器只访问一个入口地址，不依赖宿主机 Python HTTP Server。
- 前端 API、配置助手和实验监控 SSE 请求都能在容器网络中正常工作。
- 实验场景智能体可以通过容器内网络调用随部署包提供的 Qwen3-0.6B GGUF 模型。
- SQLite 数据库在容器重启后保留。
- real_data 可以作为只读数据目录挂载。
- LLM API 密钥只在运行时注入，不进入镜像或构建日志。
- 保留未配置 LLM 时的本地规则解析能力。

### 1.2 范围

本次覆盖 FastAPI 后端、静态前端、Nginx 反向代理、SQLite 持久化、real_data 挂载、本地 Qwen3-0.6B 模型容器、外部 LLM 环境变量、实验创建与监控、Docker 构建和部署文档。

本次暂不覆盖多后端高可用、SQLite 到 PostgreSQL 的迁移、独立任务队列、Kubernetes Helm Chart 和 GPU/CUDA 专用镜像。

## 2. 当前项目现状

### 2.1 后端

- 框架：FastAPI + Uvicorn。
- 依赖文件：[0623v1李-后端/requirements.txt](0623v1李-后端/requirements.txt)。
- 启动入口：[0623v1李-后端/app/main.py](0623v1李-后端/app/main.py)。
- 数据库：SQLite + aiosqlite，默认路径为 sqlite+aiosqlite:///./fedcompass.db。
- 启动时自动建表、写入静态种子数据，并处理重启时遗留的 running 实验。
- 实验运行和 SSE 事件依赖单个后端进程的内存状态。

### 2.2 前端

- 当前是无需 Node 构建的静态前端。
- 当前开发启动方式是 python -m http.server 4173。
- 前端入口：[0624v1冠-前端/FL系统/frontend/app.js](0624v1冠-前端/FL系统/frontend/app.js)。
- 当前 API_BASE 写死为 http://localhost:8000，部署到其他服务器时必须改为同源相对地址或可配置地址。
- 运行监控通过 SSE 访问后端，需要 Nginx 关闭代理缓冲并延长读取超时。

### 2.3 模型、数据与密钥

- 本地模型：Qwen3-0.6B，官方 GGUF Q8_0 量化文件，路径为 `models/qwen3-0.6b/Qwen3-0.6B-Q8_0.gguf`，约 610 MiB。
- 模型服务：由 `docker/model/Dockerfile` 基于官方 `ghcr.io/ggml-org/llama.cpp:server` 镜像构建，容器监听 8001，仅加入 Compose 内部网络。
- 模型服务默认使用 CPU 层（`--gpu-layers 0`），不依赖宿主机安装 llama.cpp，也不需要把模型权重挂载到宿主机。

- 当前有效数据库：[0623v1李-后端/fedcompass.db](0623v1李-后端/fedcompass.db)。
- real_data 目录约 205 MB，包含 CIFAR、Market-1501、ModelNet40 等数据。
- 后端通过工作区路径推导 real_data，容器内需要改为环境变量驱动。
- [.env.example](0623v1李-后端/.env.example) 定义了 LLM 配置。
- 当前本地 .env 含有实际 LLM 配置，构建上下文必须排除 .env。

本次执行已新增第一版 Dockerfile、Nginx 配置、docker-compose.yml、模型权重打包配置和部署说明；由于当前开发机未安装 Docker CLI，镜像构建和 Compose 启动尚未完成。

## 3. 推荐部署架构

采用三个容器：

    浏览器
      |
      v
    Nginx 前端容器 :8080
      |-- /        -> 静态前端资源
      |-- /api/*   -> backend:8000
      |-- SSE      -> backend:8000，关闭代理缓冲
                      |
                      |-- /data/fedcompass.db
                      |-- /data/real_data 只读数据目录
                      |-- http://model:8001/v1
                                  |
                                  |-- Qwen3-0.6B-Q8_0.gguf
                      |
                      |-- 外部 LLM API（按请求选择）

### 3.1 前端容器

- 使用 nginx:alpine。
- 复制静态前端到 Nginx 静态目录。
- 容器监听 80，宿主机默认映射为 8080。
- /api/ 代理到 backend:8000。
- 配置 SSE 长连接、关闭 proxy buffering，并设置较长的 proxy read timeout。
- 不直接把后端端口暴露给浏览器。

### 3.2 后端容器

- 使用 Python 3.11 或 Python 3.12 slim。
- 首期使用 CPU 版本 PyTorch，不绑定 CUDA。
- 监听 0.0.0.0:8000。
- 以非 root 用户运行。
- 数据库写入 /data/fedcompass.db。
- 实验数据从 /data/real_data 读取。
- 增加 /healthz 健康检查接口。
- 不使用 --reload。

### 3.3 模型容器

- 使用 `ghcr.io/ggml-org/llama.cpp:server` 作为运行时基础镜像。
- 从项目构建上下文复制 `models/qwen3-0.6b/Qwen3-0.6B-Q8_0.gguf`，模型权重随模型镜像一起分发。
- 使用模型别名 `Qwen/Qwen3-0.6B`，对外提供 OpenAI-compatible `/v1/chat/completions` 接口。
- 容器监听 8001，但 Compose 不将该端口映射到宿主机；只有 backend 通过服务名 `model` 访问。
- 当前以 `--ctx-size 8192`、`--parallel 1`、`--gpu-layers 0` 启动，避免依赖特定 GPU 运行时。
- 模型文件不是密钥，不写入数据库；如需升级模型，只需替换项目模型文件并重新构建 `model` 镜像。

### 3.4 数据卷

- fedcompass_db：持久化 SQLite 数据库。
- real_data：建议绑定宿主机目录并以只读方式挂载。
- 如需单文件分发，可额外提供把 real_data 打入镜像的构建方式。

## 4. 实施阶段

### 阶段一：统一部署配置

修改 [0623v1李-后端/app/core/config.py](0623v1李-后端/app/core/config.py)，支持以下运行时配置：

    DATABASE_URL=sqlite+aiosqlite:////data/fedcompass.db
    REAL_DATA_ROOT=/data/real_data
    CORS_ORIGINS=["http://localhost:8080"]
    LLM_API_KEY=
    LLM_BASE_URL=
    LLM_MODEL=gpt-4o-mini
    LLM_TIMEOUT_SECONDS=20

调整数据路径解析：

- 优先使用 REAL_DATA_ROOT。
- 未配置时才回退到本地工作区路径。
- 避免源码目录层级变化导致容器找不到数据。

增加 GET /healthz，至少检查应用进程和数据库连接并返回 HTTP 200。

### 阶段二：后端容器化

新增 docker/backend/Dockerfile，完成以下工作：

1. 使用 Python 3.11/3.12 slim。
2. 安装 PyTorch、Pillow、数据库驱动和其他 requirements。
3. 复制后端应用代码。
4. 创建非 root 运行用户。
5. 声明 /data 为运行时数据目录。
6. 暴露 8000 端口。
7. 使用以下命令启动：

    uvicorn app.main:app --host 0.0.0.0 --port 8000

### 阶段三：前端生产化

将固定的 http://localhost:8000 改为运行时可配置地址：

- Docker 生产环境默认使用同源相对地址。
- 本地 4173 开发模式继续访问 http://localhost:8000。
- 保留运行时 API 地址覆盖能力。

建议逻辑：

    const API_BASE = window.__FEDCOMPASS_API_BASE__ ||
      (window.location.port === "4173" ? "http://localhost:8000" : "");

新增 docker/frontend/Dockerfile 和 docker/frontend/nginx.conf，配置静态资源、/api/ 反向代理、SSE 无缓冲、长超时和前端刷新回退。

### 阶段四：模型容器化

新增 `docker/model/Dockerfile`，完成以下工作：

1. 使用官方 llama.cpp server 基础镜像。
2. 将项目目录中的 Qwen3-0.6B GGUF 权重复制到 `/models`。
3. 以 OpenAI-compatible 方式监听 8001。
4. 默认使用 CPU 推理参数，保证通用 Linux 服务器可以启动。
5. 保留 `LLAMA_IMAGE` 构建参数，便于后续切换已验证的镜像来源。

模型文件必须保留在 Docker 构建上下文中；`.dockerignore` 不得排除 `models/` 或 `*.gguf`。

### 阶段五：Docker Compose

新增 docker-compose.yml 和 .dockerignore。

Compose 需要包含 model、backend 和 frontend 三个服务：

    model:
      build:
        context: .
        dockerfile: docker/model/Dockerfile
      expose:
        - 8001

    backend:
      expose: 8000
      volumes:
        - fedcompass_db:/data
        - ./real_data:/data/real_data:ro
      healthcheck: GET /healthz
      depends_on:
        model:
          condition: service_started
      environment:
        LLM_PROVIDER: local
        LLM_LOCAL_BASE_URL: http://model:8001/v1
        LLM_LOCAL_MODEL: Qwen/Qwen3-0.6B
        LLM_MAX_TOKENS: 1024

    frontend:
      ports:
        - 8080:80
      depends_on:
        backend:
          condition: service_healthy

    volumes:
      fedcompass_db:

实际实现已补充环境变量、重启策略、容器权限和模型服务依赖。模型服务启动较慢时，智能体请求会由后端按现有逻辑回退到本地规则解析；模型就绪后自动使用 `llm_local` 路径。

### 阶段六：安全和构建上下文

.dockerignore 至少排除：

    .env
    .venv-fedcompass
    .git
    __pycache__
    *.pyc
    .DS_Store
    参考/

以下内容必须保留在构建上下文中：

    models/qwen3-0.6b/Qwen3-0.6B-Q8_0.gguf

安全要求：

- 不把真实 LLM API Key 写入 Dockerfile。
- 不把 .env 复制进镜像。
- 不在构建日志输出完整环境变量。
- 生产环境限制 CORS_ORIGINS 为实际域名。
- real_data 使用只读挂载。
- SQLite 和实验输出目录使用可写数据卷。
- 镜像以非 root 用户运行。

### 阶段七：数据库和数据迁移

新部署使用新的 fedcompass_db volume，后端启动时自动建表并写入静态 seed 数据。

迁移已有环境时，将 [0623v1李-后端/fedcompass.db](0623v1李-后端/fedcompass.db) 复制到 Docker 数据卷，启动后检查实验记录、结果和事件，不直接覆盖已有数据库。

当前项目没有 Alembic 等迁移工具。后续数据库结构频繁变化时，应增加迁移机制，避免只依赖 create_all。

## 5. 构建与部署命令

    docker compose build model backend frontend
    docker compose up -d
    docker compose ps
    docker compose logs -f backend
    docker compose logs -f frontend
    docker compose down

删除数据库卷前必须明确确认，因为该操作会丢失实验记录：

    docker compose down -v

默认访问地址：

    http://服务器IP:8080

## 6. 验收测试

### 6.1 构建验收

- 干净服务器上构建成功。
- 镜像不包含 .env 和实际 LLM 密钥。
- 镜像不依赖宿主机 Python 或 Node。
- Linux x86_64 服务器可以启动。
- `model` 镜像构建成功，并包含 Qwen3-0.6B GGUF 权重。
- `backend` 能解析 `http://model:8001/v1`，不依赖宿主机 `localhost`。
- 模型容器没有把 8001 直接暴露到宿主机公网。

### 6.2 服务验收

- GET /healthz 返回 200。
- 前端首页返回 200。
- /api/overview、/api/modules 等接口正常。
- 浏览器请求不再指向浏览器所在机器的 localhost:8000。

### 6.3 配置助手验收

- 未配置 LLM 时，本地规则解析正常。
- 配置 LLM 环境变量后，LLM 路径正常。
- LLM 密钥不出现在接口响应、前端资源和容器日志中。
- 默认本地模型请求返回 `provider=llm_local`；模型未就绪时允许受控回退为 `local_fallback`。
- 缺失字段提醒正常。
- 点击“应用配置”不会覆盖缺失字段。

### 6.4 实验流程验收

- 可以创建、启动和停止实验。
- 运行监控页面可以持续接收 SSE 事件。
- 实验完成后可以查看结果和指标。
- 后端重启后数据库中的实验记录仍然存在。
- 后端重启时遗留的 running 实验会按当前逻辑被标记为 stopped。

### 6.5 模型与数据验收

- CIFAR、Market-1501 和 ModelNet40 数据可以读取。
- 未挂载 real_data 时，系统仍可以启动并使用 mock/演示能力。
- 模型服务 `/v1/models` 返回 `Qwen/Qwen3-0.6B`。
- 模型服务 `/v1/chat/completions` 能返回 JSON 配置草案。
- 首次加载模型时，后端和 Nginx 的启动顺序不会导致不可恢复失败。

## 7. 部署边界与后续扩展

首期约束：

- 单个后端容器实例。
- SQLite 数据库。
- CPU PyTorch。
- Qwen3-0.6B GGUF 模型由独立模型容器提供，当前使用 CPU 推理。
- real_data 外部只读挂载。
- 通过 Nginx 提供统一入口。

当前实验执行状态、SSE 事件和部分运行逻辑依赖单进程内存，因此不直接支持多个后端副本。未来如需高并发或高可用，应依次考虑：

1. SQLite 迁移到 PostgreSQL。
2. 实验任务迁移到 Redis/Celery 或其他任务队列。
3. SSE 事件迁移到共享消息系统。
4. 使用对象存储保存模型、日志和结果。
5. 再考虑多后端副本和 Kubernetes 部署。

## 8. 最终交付物

已新增或纳入部署包：

    docker-compose.yml
    docker/backend/Dockerfile
    docker/frontend/Dockerfile
    docker/frontend/nginx.conf
    docker/model/Dockerfile
    models/qwen3-0.6b/Qwen3-0.6B-Q8_0.gguf
    .dockerignore
    部署说明.md

预计修改：

    0623v1李-后端/app/core/config.py
    0623v1李-后端/app/main.py
    0623v1李-后端/app/services/services.py
    0624v1冠-前端/FL系统/frontend/app.js
    0624v1冠-前端/FL系统/frontend/index.html

前后端、模型容器和 Compose 配置已经落实。当前仅剩在具备 Docker Engine 的环境中完成镜像构建、Compose 启动和跨服务器验收；本机已完成模型服务与后端接口的真实联调。
