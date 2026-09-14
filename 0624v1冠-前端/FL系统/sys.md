成果到系统模块映射
FedCVC：原始-对偶核心模块，加入客户端 silence table、client-driven virtual compensation。
FedDTC：动态异质性感知与全局梯度漂移校正模块，维护 EMA 异质性指标和历史梯度表。
RFL-NLCP：基座算法模块，保留 PCA/KMeans 客户端聚类、cluster-aware dual amplification、SAM/ESAM。
FedCADS：蒸馏增强模块，加入 self/global dual distillation、动态蒸馏权重、参与率感知对偶更新。
GFed-HSAM：局部优化器模块，抽象 SAM/ESAM/HSAM 及全局 sharpness-aware 约束。
VERT：安全防御模块，作为 defense plugin 接入聚合前的梯度筛选，支持 GN/MR/AGR/ALIE 等投毒攻击模拟。
CO-EVO：ReID/FedDG 应用插件，接入 semantic anchor、global style bank、mAP/Rank-1 评测。
FedULIP：3D 多模态应用插件，接入 ULIP backbone、adapter/SACA/inversion network、base-to-new/cross-dataset/domain ABCD 场景。


fedcompass/
  core/              # Server, Client, StateStore, Registry, Trainer
  algorithms/        # FedAvg, FedDyn, RFLNLCP, FedCVC, FedDTC, FedCADS
  optimizers/        # SGD, SAM, ESAM, HSAM
  defenses/          # VERT, Krum, Median, FLDetector, attacks
  datasets/          # IID, Dirichlet, Pathological, domain-as-client
  apps/
    vision/          # CIFAR/MNIST/AG_News
    reid/            # CO-EVO
    ulip3d/          # FedULIP
  configs/           # YAML experiment configs
  scripts/           # run sweeps, plot, summarize
  outputs/           # logs, checkpoints, tables



集成计划
第 0 阶段：资产梳理
克隆 RFL-NLCP、FedCADS、VERT、FedULIP；补齐 FedCVC/FedDTC/CO-EVO 缺失代码状态；建立论文-代码-模块映射表。

第 1 阶段：系统骨架
从 RFL-NLCP 抽出 BaseServer、BaseClient、AlgorithmPlugin、LocalOptimizer、StateStore、MetricLogger。先跑通 FedAvg、FedDyn、RFL-NLCP 三个 smoke test。

第 2 阶段：核心 FL 算法集成
先集成 FedCVC 和 FedDTC，因为它们定义系统主线：原始-对偶、低参与、动态补偿。随后接 FedCADS 的蒸馏钩子和 GFed-HSAM 的优化器钩子。

第 3 阶段：鲁棒安全层
将 VERT 改成聚合前 defense plugin：输入客户端更新和历史梯度，输出可信客户端集合或重加权更新。同步接入攻击模拟和 detection accuracy 指标。

第 4 阶段：应用扩展
CO-EVO 放到 ReID/FedDG 分支；FedULIP 放到 3D 多模态分支。二者都复用核心 server-client loop，但保留各自任务评测指标。

第 5 阶段：实验与发布
统一 YAML 配置、随机种子、日志、TensorBoard/W&B、结果表格生成。最终给出 CIFAR 基础复现实验、低参与率扫描、投毒防御实验、ReID/ULIP 应用样例。
