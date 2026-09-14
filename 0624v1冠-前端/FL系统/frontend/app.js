const API_BASE =
  typeof window !== "undefined" && window.__FEDCOMPASS_API_BASE__
    ? window.__FEDCOMPASS_API_BASE__
    : typeof window !== "undefined" && window.location.port === "4173"
      ? "http://localhost:8000"
      : "";
const BACKEND_CHECK_TTL_MS = 15000;
const API_CACHE_TTL_MS = 12000;

const routes = [
  { id: "dashboard", label: "总览" },
  { id: "modules", label: "模块能力" },
  { id: "builder", label: "实验配置" },
  { id: "monitor", label: "运行监控" },
  { id: "results", label: "结果对比" },
  { id: "apps", label: "应用场景" },
];

const SMOKE_VALIDATED_ALGORITHM_IDS = new Set([
  "fedavg",
  "feddyn",
  "fedprox",
  "scaffold",
  "fednova",
  "fedadam",
  "fedyogi",
  "fedbn",
  "ditto",
  "pfedme",
  "fedproto",
  "feddf",
  "fedkd",
  "fedcvc",
  "feddtc",
  "rfl_nlcp",
  "gfed_hsam",
  "sam",
  "esam",
  "hsam",
  "fedcads",
  "a_fedpd",
  "a_fedpdsam",
  "fedspeed",
  "fedsmoo",
  "fedlesam_d",
  "fedgloss",
  "feddc",
  "fedvra",
  "fedvarp",
  "fedtoga",
  "fedgkd_p",
  "fedfld",
  "vert",
  "krum",
  "median",
  "trimmed_mean",
  "fldetector",
  "flbeeline",
  "multi_krum",
  "fltrust",
  "flame",
  "featurepoison",
  "fedproto_prototype_attack",
  "sgd",
  "gn",
  "mr",
  "agr",
  "alie",
  "co_evo",
  "moon",
  "mixstyle",
  "crossstyle",
  "fedreid",
  "fedpav",
  "snr",
  "dacs",
  "sscu",
  "fedulip",
  "ulip",
  "pointclip",
  "fedkgcoop",
  "fedvpt",
  "fedtpg",
  "fedcocoop",
  "fedmaple",
  "fedclip",
  "fedmvp",
]);

const state = {
  route: "dashboard",
  selectedModuleId: null,
  moduleReferences: {},
  resultsMetric: "accuracy",
  resultsFilter: {},
  monitorMetric: "test_accuracy",
  builderCatalog: null,
  builderTemplates: [],
  builderConfig: {
    module_id: "vision_benchmark",
    task_type: "vision_classification",
    dataset: "cifar10",
    split: "iid",
    network: "resnet18",
    algorithm: "fedavg",
    optimizer: "sgd",
    defense: "none",
    attack: "none",
    num_clients: 30,
    participation_rate: 0.1,
    rounds: 10,
    local_epochs: 5,
    batch_size: 64,
    learning_rate: 0.01,
    seed: 42,
    trust_threshold: 0.75,
    distill_weight: 0.4,
    global_distill_weight: 0.6,
    temperature: 2.0,
  },
  selectedExperimentId: "exp_demo_001",
  backendAvailable: false,
  alert: null,
  configAssistant: {
    message: "",
    loading: false,
    draft: null,
    error: null,
    llm: {
      provider: "local",
      baseUrl: "",
      apiKey: "",
      model: "",
    },
  },
};

const mockDb = createMockDatabase();
let monitorTimer = null;
let eventSourceRef = null;
let revealedRounds = 0;
let lastMonitorExperimentId = null;
let monitorBootExperimentId = null;
let monitorBootStage = "preflight";
let selectedHeatmapRoundValue = null;
let backendCheckInFlight = null;
let backendLastCheckedAt = 0;
let routeWarmupStarted = false;
const apiResponseCache = new Map();

const navEl = document.querySelector("#nav");
const titleEl = document.querySelector("#page-title");
const rootEl = document.querySelector("#view-root");
const alertRegion = document.querySelector("#alert-region");
const refreshButton = document.querySelector("#refresh-button");
const mockToggleButton = document.querySelector("#mock-toggle");
const backendStatus = document.querySelector("#backend-status");
const apiBaseLabel = document.querySelector("#api-base-label");

renderNav();
bindGlobalEvents();
checkBackend({ force: true })
  .then(() => renderCurrentRoute())
  .then(scheduleRouteWarmup);

function bindGlobalEvents() {
  refreshButton.addEventListener("click", () => {
    clearApiCache();
    renderCurrentRoute({ refreshBackend: true });
  });
  if (mockToggleButton) {
    mockToggleButton.addEventListener("click", () => {
      showAlert("info", "当前已统一为自动模式：有真实后端数据就展示真实数据，后端不可用时自动回退到模拟数据。");
    });
  }
  window.addEventListener("hashchange", () => {
    const hashRoute = location.hash.replace("#", "");
    if (routes.some((item) => item.id === hashRoute)) {
      state.route = hashRoute;
      renderNav();
      renderCurrentRoute();
    }
  });
}

function navigateToRoute(route) {
  if (!routes.some((item) => item.id === route)) {
    return;
  }
  state.route = route;
  renderNav();
  if (location.hash.replace("#", "") !== route) {
    location.hash = route;
    return;
  }
  renderCurrentRoute();
}

function renderNav() {
  const current = location.hash.replace("#", "");
  if (routes.some((item) => item.id === current)) {
    state.route = current;
  }
  navEl.innerHTML = routes
    .map(
      (item) => `
        <button class="nav-button ${state.route === item.id ? "active" : ""}" data-route="${item.id}">
          ${item.label}
        </button>
      `
    )
    .join("");

  navEl.querySelectorAll("[data-route]").forEach((button) => {
    button.addEventListener("click", () => {
      navigateToRoute(button.dataset.route);
    });
  });
}

async function checkBackend({ force = false } = {}) {
  apiBaseLabel.textContent = "自动模式：优先真实数据，失败回退模拟数据";
  const now = Date.now();
  if (!force && backendLastCheckedAt && now - backendLastCheckedAt < BACKEND_CHECK_TTL_MS) {
    syncBackendBadge();
    return state.backendAvailable;
  }
  if (backendCheckInFlight) {
    return backendCheckInFlight;
  }
  backendCheckInFlight = (async () => {
    try {
      const response = await fetch(`${API_BASE}/api/overview`, { method: "GET" });
      state.backendAvailable = response.ok;
    } catch (error) {
      state.backendAvailable = false;
    } finally {
      backendLastCheckedAt = Date.now();
      syncBackendBadge();
      backendCheckInFlight = null;
    }
    return state.backendAvailable;
  })();
  return backendCheckInFlight;
}

function syncBackendBadge() {
  if (state.backendAvailable) {
    backendStatus.textContent = "真实数据在线";
    backendStatus.className = "status-badge ok";
  } else {
    backendStatus.textContent = "后端暂不可达，已回退模拟数据";
    backendStatus.className = "status-badge warn";
  }
  if (mockToggleButton) {
    mockToggleButton.textContent = "自动模式";
    mockToggleButton.disabled = true;
    mockToggleButton.title = "统一自动模式：优先真实数据，失败回退模拟数据";
  }
}

function showAlert(type, message) {
  state.alert = { type, message };
  renderAlert();
}

function renderAlert() {
  if (!state.alert) {
    alertRegion.innerHTML = "";
    return;
  }
  alertRegion.innerHTML = `<div class="alert ${state.alert.type}">${escapeHtml(state.alert.message)}</div>`;
}

function clearAlert() {
  state.alert = null;
  renderAlert();
}

async function renderCurrentRoute({ refreshBackend = false } = {}) {
  clearMonitorResources();
  clearAlert();
  checkBackend({ force: refreshBackend });
  titleEl.textContent = routes.find((item) => item.id === state.route)?.label || "总览";
  renderRouteLoading(state.route);

  switch (state.route) {
    case "dashboard":
      await renderDashboard();
      break;
    case "modules":
      await renderModules();
      break;
    case "builder":
      await renderBuilder();
      break;
    case "monitor":
      await renderMonitor();
      break;
    case "results":
      await renderResults();
      break;
    case "apps":
      await renderApps();
      break;
    default:
      rootEl.innerHTML = "";
  }
  scheduleRouteWarmup();
}

function renderRouteLoading(route) {
  const label = routes.find((item) => item.id === route)?.label || "总览";
  rootEl.innerHTML = `
    <section class="route-loading panel" aria-live="polite">
      <div>
        <p class="eyebrow">FedCompass</p>
        <h3>${escapeHtml(label)}加载中</h3>
      </div>
      <div class="route-loading-bars" aria-hidden="true">
        <span></span><span></span><span></span>
      </div>
    </section>
  `;
}

function scheduleRouteWarmup() {
  if (routeWarmupStarted || !state.backendAvailable) {
    return;
  }
  routeWarmupStarted = true;
  window.setTimeout(() => {
    Promise.allSettled([
      api.getModules(),
      api.getAlgorithms(),
      api.getDatasets(),
      api.getOptimizers(),
      api.getDefenses(),
      api.getAttacks(),
      api.getExperimentTemplates(),
      api.getExperimentConfigOptions("vision_benchmark"),
      api.getExperimentConfigOptions("reid_generalization"),
      api.getExperimentConfigOptions("multimodal_3d"),
      api.getNetworks("vision_classification"),
      api.getNetworks("reid"),
      api.getNetworks("ulip3d"),
      api.getModuleById("vision_benchmark"),
      api.getModuleById("reid_generalization"),
      api.getModuleById("multimodal_3d"),
    ]);
  }, 300);
}

async function renderDashboard() {
  const [overview, capabilityMap, runningExperiments, recentResults] = await Promise.all([
    api.getOverview(),
    api.getOverviewCapabilityMap(),
    api.getExperiments({ status: "running" }),
    api.getRecentResults(10),
  ]);
  if (state.route !== "dashboard") {
    return;
  }

  rootEl.innerHTML = `
    <section class="showcase-hero">
      <div class="showcase-copy">
        <p class="eyebrow">FedCompass Live Lab</p>
        <h3>FedCompass</h3>
        <p>
          三类应用场景、${overview.algorithm_count} 个算法方案、真实后端小批量运行，
          从任务选择到训练监控再到结果对比，把系统能力直接呈现在观众面前。
        </p>
        <div class="showcase-impact-row">
          ${[
            [String(overview.algorithm_count), "注册算法"],
            [String(overview.dataset_count), "可选数据集"],
            ["3", "应用场景"],
            ["4", "演示证据层"],
          ]
            .map(
              ([value, label]) => `
                <div class="showcase-impact-card">
                  <strong>${escapeHtml(value)}</strong>
                  <span>${escapeHtml(label)}</span>
                </div>
              `
            )
            .join("")}
        </div>
        <div class="showcase-actions">
          <button class="primary-button" type="button" data-app-route="apps">查看应用场景</button>
          <button class="secondary-button" type="button" data-app-route="builder">启动演示实验</button>
        </div>
      </div>
      <div class="showcase-command-center" aria-label="FedCompass 实验驾驶舱">
        <div class="command-topline">
          <span>Live Federated Run</span>
          <strong>ONLINE</strong>
        </div>
        <div class="command-network">
          <div class="command-hub">
            <span>FC</span>
          </div>
          <span class="command-node node-vision">Vision</span>
          <span class="command-node node-reid">ReID</span>
          <span class="command-node node-3d">3D</span>
          <span class="command-node node-defense">Robust</span>
          <span class="command-link link-a"></span>
          <span class="command-link link-b"></span>
          <span class="command-link link-c"></span>
          <span class="command-link link-d"></span>
        </div>
        <div class="command-evidence-strip">
          ${[
            ["场景设置", "锁定"],
            ["客户端参与", "可视"],
            ["训练指标", "同步"],
            ["基线对比", "完成"],
          ]
            .map(
              ([label, value]) => `
                <div>
                  <span>${escapeHtml(label)}</span>
                  <strong>${escapeHtml(value)}</strong>
                </div>
              `
            )
            .join("")}
        </div>
        <div class="command-signal">
          <span style="height:38%"></span>
          <span style="height:56%"></span>
          <span style="height:44%"></span>
          <span style="height:72%"></span>
          <span style="height:63%"></span>
          <span style="height:86%"></span>
          <span style="height:78%"></span>
        </div>
      </div>
    </section>

    <section class="grid cols-4">
      ${renderMetricCard("运行中任务", overview.running_experiments, "当前后台活跃的实验任务数")}
      ${renderMetricCard("已注册算法", overview.algorithm_count, "算法与防御策略总量")}
      ${renderMetricCard("已注册数据集", overview.dataset_count, "当前可选任务数据")}
      ${renderMetricCard("可用防御策略", overview.defense_count, "面向鲁棒训练与检测")}
    </section>

    <section class="dashboard-main">
      <div class="panel dashboard-map-panel">
        <h3>模块能力地图</h3>
        <p class="muted">聚合展示 11 个模块的定位、状态与能力标签。</p>
        <div class="matrix">
          ${capabilityMap
            .map(
              (module) => `
                <article class="matrix-card">
                  <h4>${escapeHtml(module.name)}</h4>
                  <p class="muted">${escapeHtml(module.positioning)}</p>
                  <div class="pill-row">
                    ${(module.tags || []).map((tag) => `<span class="pill">${escapeHtml(tag)}</span>`).join("")}
                  </div>
                </article>
              `
            )
            .join("")}
        </div>
      </div>

      <div class="dashboard-side">
        <div class="panel dashboard-side-panel">
          <div class="dashboard-side-content dashboard-results-content dashboard-content-top">
            <h3>近期结果摘要</h3>
            <p class="muted">用柱状图展示近期实验的最终精度表现。</p>
            ${renderBarChart(recentResults, "final_accuracy", "experiment_name")}
          </div>
        </div>

        <div class="panel dashboard-side-panel">
          <div class="dashboard-side-content dashboard-running-content">
            <h3>运行中实验</h3>
            <p class="muted">展示当前正在运行的实验任务与关键进度。</p>
            <div class="list">
              ${runningExperiments.map(renderRunningExperiment).join("")}
            </div>
          </div>
        </div>

        <div class="panel dashboard-side-panel">
          <div class="dashboard-side-content dashboard-content-top">
            <h3>关键指标总览</h3>
            <p class="muted">汇总展示系统当前的核心实验指标。</p>
            <div class="list">
              ${[
                ["精度", `${formatPercent(overview.key_metrics.accuracy)}`],
                ["mAP", `${formatPercent(overview.key_metrics.map)}`],
                ["Rank-1", `${formatPercent(overview.key_metrics.rank1)}`],
                ["攻击检测精度", `${formatPercent(overview.key_metrics.attack_detection_accuracy)}`],
                ["通信开销", `${overview.key_metrics.communication_cost} MB`],
              ]
                .map(
                  ([label, value]) => `
                    <div class="list-item">
                      <strong>${escapeHtml(label)}</strong>
                      <span class="metric-value">${escapeHtml(value)}</span>
                    </div>
                  `
                )
                .join("")}
            </div>
          </div>
        </div>
      </div>
    </section>
  `;
  bindRouteActions();
}

async function renderModules() {
  const modules = await api.getModules();
  if (state.route !== "modules") {
    return;
  }
  if (!state.selectedModuleId) {
    state.selectedModuleId = modules[0]?.id || null;
  }
  const selectedModule = modules.find((item) => item.id === state.selectedModuleId) || modules[0];
  const selectedModuleStatus = getModuleDisplayStatus(selectedModule);
  const moduleValidatedCount = modules.filter((module) => getModuleDisplayStatus(module) === "validated").length;

  rootEl.innerHTML = `
    ${renderPageShowcase("modules", "联邦能力矩阵，一眼看清成果版图", "把训练核心、鲁棒防护、优化控制和三类应用场景放在同一张能力地图里，展示系统已经具备的算法广度和接入成熟度。", [
      { label: "能力模块", value: `${modules.length} 个` },
      { label: "已验证模块", value: `${moduleValidatedCount} 个` },
      { label: "当前算法", value: `${(selectedModule.algorithms || []).length} 项` },
      { label: "当前状态", value: selectedModuleStatus === "validated" ? "已验证" : formatStatus(selectedModuleStatus) },
    ])}
    <section class="split-layout">
      <aside class="panel">
        <h3>模块列表</h3>
        <div class="module-list">
          ${modules
            .map(
              (module) => `
                <button class="module-button ${module.id === selectedModule.id ? "active" : ""}" data-module-id="${module.id}">
                  <strong>${escapeHtml(module.name)}</strong>
                  <p class="muted subtle">${escapeHtml(module.positioning)}</p>
                </button>
              `
            )
            .join("")}
        </div>
      </aside>

      <section class="panel detail-block">
        <div class="module-hero">
          <div class="pill-row">
            ${(selectedModule.frontend_views || []).map((view) => `<span class="pill">${escapeHtml(view)}</span>`).join("")}
          </div>
          <h3>${escapeHtml(selectedModule.name)}</h3>
          <p>${escapeHtml(selectedModule.description)}</p>
        </div>

        <div>
          <h4>算法与参考基线</h4>
          ${renderModuleAlgorithmGroups(selectedModule)}
        </div>

        <div>
          <h4>页面展示重点</h4>
          <div class="list">
            ${(selectedModule.showcase_points || []).map((point) => `<div class="list-item"><p>${escapeHtml(point)}</p></div>`).join("")}
          </div>
        </div>

        <div>
          <h4>解决问题</h4>
          <p class="muted">${escapeHtml(selectedModule.problem_focus)}</p>
        </div>

        <div>
          <h4>输入输出</h4>
          <p class="muted">${escapeHtml(selectedModule.io_contract)}</p>
        </div>

        <div>
          <h4>接口示意</h4>
          <div class="list">
            <div class="list-item">
              <strong>后端服务</strong>
              <p>${escapeHtml(selectedModule.backend_service || "模块服务待接入")}</p>
            </div>
            <div class="list-item">
              <strong>Plugin Hook</strong>
              <p>${escapeHtml((selectedModule.backend_plugins || []).join(" / "))}</p>
            </div>
            <div class="list-item">
              <strong>配置字段</strong>
              <p>${escapeHtml((selectedModule.config_fields || []).join(" / "))}</p>
            </div>
          </div>
        </div>
      </section>

      <aside class="panel detail-block compact-panel">
        <div>
          <h3>完成标记</h3>
          <p class="muted">已完成阶段高亮显示。</p>
        </div>
        <div class="stage-list" id="stage-list">
          ${renderStageProgression(selectedModuleStatus)}
        </div>
      </aside>
    </section>
  `;

  rootEl.querySelectorAll("[data-module-id]").forEach((button) => {
    button.addEventListener("click", async () => {
      state.selectedModuleId = button.dataset.moduleId;
      await renderModules();
    });
  });


}


function getModuleDisplayStatus(module) {
  const algorithms = module?.algorithms || [];
  const validatedAlgorithms = algorithms.filter((algorithm) => algorithm.status === "validated").length;
  if (algorithms.length > 0 && validatedAlgorithms === algorithms.length) {
    return "validated";
  }
  return module?.implementation_status || module?.status || "paper_only";
}

function renderModuleAlgorithmGroups(module) {
  const algorithms = module.algorithms || [];
  const byId = new Map(algorithms.map((algorithm) => [algorithm.id, algorithm]));
  const primaryIds = uniqueValues(module.primary_algorithms || []);
  const primarySet = new Set(primaryIds);
  const baselineIds = uniqueValues(module.baseline_algorithms || []).filter((id) => !primarySet.has(id));
  const fallbackIds = uniqueValues(algorithms.map((algorithm) => algorithm.id));
  const effectivePrimaryIds = primaryIds.length ? primaryIds : fallbackIds;

  const renderGroup = (title, ids, kind) => {
    const pills = ids.map((id) => renderModuleAlgorithmPill(resolveAlgorithmBrief(id, byId), kind)).join("");
    return `
      <div class="algorithm-group ${kind === "baseline" ? "baseline-group" : ""}">
        <div class="algorithm-group-heading">
          <strong>${escapeHtml(title)}</strong>
          <span>${ids.length} 项</span>
        </div>
        <div class="pill-row">${pills || `<span class="muted subtle">暂无</span>`}</div>
      </div>
    `;
  };

  return `
    <div class="algorithm-groups">
      ${renderGroup("主算法", effectivePrimaryIds, "primary")}
      ${baselineIds.length ? renderGroup("参考基线", baselineIds, "baseline") : ""}
    </div>
  `;
}

function resolveAlgorithmBrief(id, byId) {
  return byId.get(id) || {
    id,
    name: formatAlgorithmDisplay(id),
    category: "paper_baseline",
    status: "paper_only",
  };
}

function renderModuleAlgorithmPill(algorithm, kind) {
  const statusClass = algorithm.status ? escapeClassName(algorithm.status) : "unknown";
  const title = [algorithm.category, algorithm.status].filter(Boolean).join(" / ");
  return `<span class="pill algorithm-pill ${kind === "baseline" ? "baseline-pill" : ""} ${statusClass}" title="${escapeHtml(title)}">${escapeHtml(algorithm.name)}</span>`;
}

function uniqueValues(values) {
  return Array.from(new Set((values || []).filter(Boolean)));
}


function renderStageProgression(currentStatus) {
  const allStages = [
    { id: "paper_only", label: "资料阶段", desc: "只有参考资料或方案" },
    { id: "external_code", label: "外部代码", desc: "代码在外部仓库" },
    { id: "local_archive", label: "本地存档", desc: "代码以本地压缩包存在" },
    { id: "mocked", label: "模拟数据", desc: "模拟数据" },
    { id: "replay_ready", label: "可回放", desc: "可读取已有日志回放" },
    { id: "implemented", label: "已实现", desc: "已完成代码接入" },
    { id: "validated", label: "已验证", desc: "已完成 smoke 验证" },
  ];
  const completedIdx = allStages.findIndex((s) => s.id === currentStatus);
  return allStages
    .map((s, i) => {
      const isCompleted = i <= completedIdx;
      return `<div class="stage-item${isCompleted ? " stage-completed" : ""}">
        <div class="stage-dot${isCompleted ? " stage-dot-active" : ""}"></div>
        <div class="stage-info">
          <span class="stage-label">${s.label}</span>
          <span class="stage-desc">${s.desc}</span>
        </div>
      </div>`;
    })
    .join("");
}

function renderReferencePanel(moduleId) {
  const panel = rootEl.querySelector("#reference-panel");
  const refs = state.moduleReferences[moduleId];
  if (!refs) {
    panel.innerHTML = renderEmptyState("点击按钮后加载参考资料、代码与接入状态。");
    return;
  }

  panel.innerHTML = `
      <div class="list">
      <div class="list-item">
        <strong>参考资料</strong>
        ${(refs.papers || []).map((paper) => `<p>${escapeHtml(paper)}</p>`).join("")}
      </div>
      <div class="list-item">
        <strong>代码来源</strong>
        ${(refs.code_sources || []).map((code) => `<p>${escapeHtml(code)}</p>`).join("")}
      </div>
      <div class="list-item">
        <strong>接入阶段</strong>
        <p>${escapeHtml(refs.implementation_status || "-")}</p>
      </div>
      <div class="list-item">
        <strong>展示页面</strong>
        ${(refs.frontend_views || []).map((view) => `<p>${escapeHtml(view)}</p>`).join("")}
      </div>
      <div class="list-item">
        <strong>关联能力</strong>
        ${(refs.backend_plugins || []).map((plugin) => `<p>${escapeHtml(plugin)}</p>`).join("")}
      </div>
    </div>
  `;
}

const TASK_TYPE_LABELS = {
  vision_classification: "视觉分类",
  text_classification: "文本分类",
  defense_demo: "攻防演示",
  reid: "ReID 域泛化",
  ulip3d: "3D 多模态",
};

const SPLIT_LABELS = {
  iid: "IID",
  dirichlet: "Dirichlet",
  pathological: "Pathological",
  domain_as_client: "Domain-as-client",
};

const PROTOCOL_LABELS = {
  visual_classification: "视觉分类",
  limited_participation: "低参与率",
  low_participation_distillation: "低参与蒸馏",
  sam_comparison: "SAM 对比",
  model_poisoning: "模型投毒",
  topk_trust_screening: "Top-K 可信筛选",
  feature_poisoning: "特征投毒",
  base2new: "base-to-new",
  cross_dataset: "cross-dataset",
  domain_abcd: "domain A/B/C/D",
  leave_one_domain_out: "Protocol I",
  multi_source_mixed_test: "Protocol II",
  source_domain_evaluation: "Protocol III",
};

const OPTION_LABELS = {
  none: "None",
  sgd: "SGD",
  small_cnn: "Small CNN",
  tiny_reid: "Tiny ReID",
  textcnn: "TextCNN",
  pointbert: "PointBERT",
  resnet18: "ResNet-18",
  resnet50: "ResNet-50",
  mobilenetv2: "MobileNetV2",
  lenet5: "LeNet-5",
  vgg16: "VGG-16",
  tiny_imagenet: "Tiny-ImageNet",
  fashion_mnist: "FashionMNIST",
  femnist: "FEMNIST",
  emnist: "EMNIST",
  ag_news: "AG News",
  cuhk02: "CUHK02",
  cuhk03: "CUHK03",
  msmt17: "MSMT17",
  shapenetcore: "ShapeNetCore",
  mvtec3d: "MVTec3D",
  mnist3d: "MNIST3D",
  "3dimage": "3Dimage",
};

function pairAllowedOptions(allowedIds = [], catalog = [], labelMap = {}) {
  const catalogMap = new Map((catalog || []).map((item) => [item.id, item.name || item.id]));
  return (allowedIds || []).map((id) => [id, catalogMap.get(id) || labelMap[id] || OPTION_LABELS[id] || id]);
}

function uniqueById(items = []) {
  const seen = new Set();
  return items.filter((item) => {
    if (!item || seen.has(item.id)) return false;
    seen.add(item.id);
    return true;
  });
}

function coerceBuilderConfigToOptions(config, options) {
  const defaults = options.defaults || {};
  config.module_id = options.module_id || config.module_id || defaults.module_id || "vision_benchmark";
  const fields = {
    task_type: "task_types",
    dataset: "datasets",
    split: "splits",
    protocol: "protocols",
    network: "networks",
    algorithm: "algorithms",
    optimizer: "optimizers",
    defense: "defenses",
    attack: "attacks",
  };
  Object.entries(fields).forEach(([field, optionKey]) => {
    const allowed = options[optionKey] || [];
    if (allowed.length && !allowed.includes(config[field])) {
      config[field] = defaults[field] || allowed[0];
    }
  });
  if (config.dataset === "ag_news" && (options.task_types || []).includes("text_classification")) {
    config.task_type = "text_classification";
    if ((options.networks || []).includes("textcnn")) config.network = "textcnn";
  } else if (config.task_type === "text_classification" && (options.datasets || []).includes("ag_news")) {
    config.dataset = "ag_news";
    if ((options.networks || []).includes("textcnn")) config.network = "textcnn";
  }
  if (config.task_type === "ulip3d") config.network = "pointbert";
  if (config.task_type === "reid" && !(options.networks || []).includes(config.network)) {
    config.network = (options.networks || [defaults.network || "tiny_reid"])[0];
  }
}

function formatAssistantValue(value) {
  if (value === undefined || value === null || value === "") {
    return "未设置";
  }
  if (typeof value === "object") {
    return JSON.stringify(value);
  }
  return String(value);
}

function assistantProviderLabel(provider) {
  if (provider === "llm_local") return "本地模型";
  if (provider === "llm_api" || provider === "llm") return "API 模型";
  return "本地规则兜底";
}

const ASSISTANT_VISIBLE_FIELDS = new Set([
  "module_id",
  "task_type",
  "dataset",
  "split",
  "protocol",
  "network",
  "algorithm",
  "optimizer",
  "defense",
  "attack",
  "num_clients",
  "participation_rate",
  "rounds",
  "local_epochs",
  "batch_size",
  "learning_rate",
  "seed",
  "trust_threshold",
  "distill_weight",
  "global_distill_weight",
  "temperature",
]);

function containsChineseText(value) {
  return /[\u4e00-\u9fff]/.test(String(value || ""));
}

function normalizeAssistantDraftLanguage(draft) {
  if (!draft) return draft;
  const config = Object.fromEntries(
    Object.entries(draft.config || {}).filter(([field]) => ASSISTANT_VISIBLE_FIELDS.has(field))
  );
  const changes = (draft.changes || []).filter((change) => ASSISTANT_VISIBLE_FIELDS.has(change.field));
  const missingFields = (draft.missing_fields || []).filter((field) => ASSISTANT_VISIBLE_FIELDS.has(field));
  const explicitFields = (draft.explicit_fields || []).filter((field) => ASSISTANT_VISIBLE_FIELDS.has(field));
  const suggestedFields = (draft.suggested_fields || []).filter((field) => ASSISTANT_VISIBLE_FIELDS.has(field));
  const recommendedFields = (draft.recommended_fields || []).filter((field) => ASSISTANT_VISIBLE_FIELDS.has(field));
  const assumptions = (draft.assumptions || [])
    .filter((note) => !note.field || ASSISTANT_VISIBLE_FIELDS.has(note.field))
    .map((note) => ({
    ...note,
    reason: containsChineseText(note.reason)
      ? note.reason
      : `本次输入未明确${FIELD_LABELS[note.field] || "相关配置项"}，请确认当前值。`,
    }));
  const warnings = (draft.warnings || []).map((warning) => (
    containsChineseText(warning)
      ? warning
      : "模型返回了一条非中文提醒，请检查配置草案中的相关参数后再确认。"
  ));
  let followUpQuestion = draft.follow_up_question || null;
  if (followUpQuestion && !containsChineseText(followUpQuestion)) {
    const labels = missingFields.map((field) => FIELD_LABELS[field] || field).join("、");
    followUpQuestion = labels ? `请确认以下配置项：${labels}。` : "请确认配置草案中的相关参数。";
  }
  return {
    ...draft,
    config,
    changes,
    missing_fields: missingFields,
    explicit_fields: explicitFields,
    suggested_fields: suggestedFields,
    recommended_fields: recommendedFields,
    mode: draft.mode || "detailed_configuration",
    assumptions,
    warnings,
    follow_up_question: followUpQuestion,
  };
}

function renderConfigAssistantPanel() {
  const assistant = state.configAssistant;
  const draft = assistant.draft;
  const llm = assistant.llm || {
    provider: "local",
    baseUrl: "",
    apiKey: "",
    model: "",
  };
  const llmProvider = llm.provider === "api" ? "api" : "local";
  const missingFields = draft?.missing_fields || [];
  const recommendedFields = draft?.recommended_fields || [];
  const suggestedFields = new Set(draft?.suggested_fields || []);
  const draftConfig = draft?.config || null;
  return `
    <section class="config-assistant-panel">
      <div class="config-assistant-header">
        <div>
          <p class="eyebrow">Experiment Scenario Agent</p>
          <h3>实验场景智能体</h3>
        </div>
        ${assistant.loading || draft ? `<span class="assistant-status ${assistant.loading ? "loading" : "ready"}">
          ${assistant.loading ? "分析场景中" : `${draft.mode === "scenario_recommendation" ? "场景建议" : "字段解析"} · ${assistantProviderLabel(draft.provider)} · ${Math.round((draft.confidence || 0) * 100)}%`}
        </span>` : ""}
      </div>
      <form id="config-assistant-form" class="config-assistant-form">
        <div class="assistant-model-config">
          <div class="assistant-model-config-header">
            <strong>模型来源</strong>
          </div>
          <div class="assistant-provider-switch" role="group" aria-label="模型来源">
            <button type="button" class="assistant-provider-option ${llmProvider === "local" ? "active" : ""}" data-assistant-provider="local">本地模型</button>
            <button type="button" class="assistant-provider-option ${llmProvider === "api" ? "active" : ""}" data-assistant-provider="api">API 模型</button>
          </div>
          ${llmProvider === "local" ? `
            <div class="assistant-local-model-notice">
              <span>本地模型使用</span>
              <strong>Qwen-0.6B</strong>
            </div>
          ` : `
            <div class="assistant-model-fields">
              <label class="assistant-model-field">
                <span>API Base URL</span>
                <input type="text" data-assistant-llm-field="baseUrl" value="${escapeHtml(llm.baseUrl || "")}" placeholder="https://api.example.com/v1" autocomplete="off">
              </label>
              <label class="assistant-model-field">
                <span>API Key</span>
                <input type="password" data-assistant-llm-field="apiKey" value="${escapeHtml(llm.apiKey || "")}" placeholder="请输入你的 API Key" autocomplete="off">
              </label>
              <label class="assistant-model-field assistant-model-field-wide">
                <span>模型名称</span>
                <input type="text" data-assistant-llm-field="model" value="${escapeHtml(llm.model || "")}" placeholder="例如：deepseek-chat / qwen-plus" autocomplete="off">
              </label>
            </div>
            <p class="assistant-model-hint">后端会以 OpenAI 兼容的 /chat/completions 协议调用你填写的服务。</p>
          `}
        </div>
        <textarea id="config-assistant-message" rows="4" placeholder="例如：我想跑一个 ReID 实验，用 Market1501 数据集，留一域协议。">${escapeHtml(assistant.message)}</textarea>
        <div class="config-assistant-examples">
          <button type="button" class="assistant-example" data-assistant-example="我想做一个视觉分类实验，用 CIFAR-10，低参与异构场景。">视觉分类示例</button>
          <button type="button" class="assistant-example" data-assistant-example="我想跑一个 ReID 实验，用 Market1501 数据集，留一域协议。">ReID 示例</button>
          <button type="button" class="assistant-example" data-assistant-example="我想跑一个 3D 多模态实验，用 ModelNet40，做 base-to-new 评测。">3D 示例</button>
        </div>
        <div class="actions">
          <button type="submit" class="primary-button" ${assistant.loading ? "disabled" : ""}>${assistant.loading ? "正在分析场景" : "生成场景配置建议"}</button>
          ${draft ? '<button type="button" class="secondary-button" id="clear-assistant-draft">清除草案</button>' : ""}
        </div>
      </form>
      ${assistant.error ? `<div class="assistant-error">${escapeHtml(assistant.error)}</div>` : ""}
      ${draft ? `
        <div class="config-assistant-draft">
          <div class="config-assistant-draft-header">
            <div>
              <h4>配置草案</h4>
              <p class="muted">请检查场景建议，确认后再应用到下方表单。</p>
            </div>
            <button type="button" class="primary-button" id="apply-assistant-config">应用配置</button>
          </div>
          ${recommendedFields.length || missingFields.length ? `
            <div class="assistant-detail-block assistant-recommendation-block">
              <strong>场景配置建议</strong>
              ${recommendedFields.length ? `
                <div class="assistant-recommendation-list">
                  ${recommendedFields.map((field) => `<div class="assistant-recommendation"><span>${escapeHtml(FIELD_LABELS[field] || field)}</span><code>${escapeHtml(formatAssistantValue(draftConfig[field]))}</code><small>${suggestedFields.has(field) ? "场景建议" : "用户输入"}</small></div>`).join("")}
                </div>
              ` : ""}
              ${missingFields.length ? `
                <div class="assistant-missing">
                  <strong>输入中未明确的配置项</strong>
                  <ul class="assistant-note-list">${missingFields.map((field) => `<li><b>${escapeHtml(FIELD_LABELS[field] || field)}</b></li>`).join("")}</ul>
                </div>
              ` : ""}
            </div>
          ` : ""}
        </div>
      ` : ""}
    </section>
  `;
}

const FIELD_LABELS = {
  module_id: "功能模块",
  task_type: "任务类型",
  dataset: "数据集",
  split: "数据划分",
  protocol: "评测协议",
  network: "网络选择",
  algorithm: "算法方案",
  optimizer: "优化器",
  defense: "防御策略",
  attack: "攻击方式",
  num_clients: "客户端数量",
  participation_rate: "每轮参与率",
  rounds: "训练轮数",
  local_epochs: "本地 Epoch",
  batch_size: "批大小",
  learning_rate: "学习率",
  seed: "随机种子",
  trust_threshold: "可信筛选阈值",
  distill_weight: "本地蒸馏权重",
  global_distill_weight: "全局蒸馏权重",
  temperature: "蒸馏温度",
};

async function renderBuilder() {
  const [modules, algorithms, datasets, optimizers, defenses, attacks, templates] = await Promise.all([
    api.getModules(),
    api.getAlgorithms(),
    api.getDatasets(),
    api.getOptimizers(),
    api.getDefenses(),
    api.getAttacks(),
    api.getExperimentTemplates(),
  ]);

  if (!state.builderConfig.module_id) {
    state.builderConfig.module_id = modules[0]?.id || "vision_benchmark";
  }

  const configOptions = await api.getExperimentConfigOptions(state.builderConfig.module_id);
  coerceBuilderConfigToOptions(state.builderConfig, configOptions);
  const networks = await api.getNetworks(state.builderConfig.task_type);
  if (state.route !== "builder") {
    return;
  }

  const currentModule = modules.find((item) => item.id === state.builderConfig.module_id) || modules[0];
  const moduleAlgorithms = algorithms.filter((item) => (item.module_ids || []).includes(currentModule?.id));
  const allowedAlgorithms = pairAllowedOptions(configOptions.algorithms, moduleAlgorithms.length ? moduleAlgorithms : algorithms);
  const allowedDatasets = pairAllowedOptions(configOptions.datasets, datasets);
  const allowedNetworks = pairAllowedOptions(configOptions.networks, networks);
  const allowedOptimizers = pairAllowedOptions(
    configOptions.optimizers,
    uniqueById([{ id: "sgd", name: "SGD" }, ...optimizers])
  );
  const allowedDefenses = pairAllowedOptions(configOptions.defenses, uniqueById([{ id: "none", name: "None" }, ...defenses]));
  const allowedAttacks = pairAllowedOptions(configOptions.attacks, uniqueById([{ id: "none", name: "None" }, ...attacks]));
  const allowedTaskTypes = (configOptions.task_types || []).map((id) => [id, TASK_TYPE_LABELS[id] || id]);
  const allowedSplits = (configOptions.splits || []).map((id) => [id, SPLIT_LABELS[id] || id]);
  const allowedProtocols = (configOptions.protocols || []).map((id) => [id, PROTOCOL_LABELS[id] || id]);
  const showDefenseAttack = allowedDefenses.some(([id]) => id !== "none") || allowedAttacks.some(([id]) => id !== "none");
  const showAttack = showDefenseAttack && allowedAttacks.length > 1;
  const showTrustParams = ["vert", "flbeeline"].includes(state.builderConfig.defense);
  const showDistillParams = state.builderConfig.module_id === "distillation_alignment" || state.builderConfig.algorithm === "fedcads";

  state.builderCatalog = { modules, algorithms, datasets, optimizers, defenses, attacks, networks, configOptions };
  state.builderTemplates = templates;

  rootEl.innerHTML = `
    ${renderPageShowcase("builder", "从场景选择到可运行实验，先完成一次运行预检", "选择功能模块后，任务类型、数据集、划分协议和算法方案会自动收敛到当前场景范围；预检通过后，再由你确认启动正式实验。", [
      { label: "当前模块", value: currentModule?.name || "未选择" },
      { label: "可选算法", value: `${allowedAlgorithms.length} 项` },
      { label: "数据集", value: `${allowedDatasets.length} 个` },
      { label: "协议", value: allowedProtocols.length ? `${allowedProtocols.length} 类` : "标准" },
    ])}
    ${renderConfigAssistantPanel()}
    <section class="builder-layout">
      <div class="panel">
        <h3>实验配置</h3>
        <p class="muted">通过表单组合一个可运行的演示实验。</p>
        <form id="builder-form" class="form-grid">
          ${renderSelectField("module_id", "功能模块", modules.map((item) => [item.id, item.name]), state.builderConfig.module_id)}
          ${renderSelectField("task_type", "任务类型", allowedTaskTypes, state.builderConfig.task_type)}
          ${renderSelectField("dataset", "数据集", allowedDatasets, state.builderConfig.dataset)}
          ${renderSelectField("split", "数据划分", allowedSplits, state.builderConfig.split)}
          ${allowedProtocols.length ? renderSelectField("protocol", "评测协议", allowedProtocols, state.builderConfig.protocol) : ""}
          ${renderSelectField("network", "网络选择", allowedNetworks, state.builderConfig.network)}
          ${renderSelectField("algorithm", "算法方案", allowedAlgorithms, state.builderConfig.algorithm)}
          ${renderSelectField("optimizer", "优化器", allowedOptimizers, state.builderConfig.optimizer)}
          ${showDefenseAttack ? renderSelectField("defense", "防御策略", allowedDefenses, state.builderConfig.defense) : ""}
          ${showAttack ? renderSelectField("attack", "攻击方式", allowedAttacks, state.builderConfig.attack) : ""}
          ${renderNumberField("num_clients", "客户端数量", state.builderConfig.num_clients, 1)}
          ${renderNumberField("participation_rate", "每轮参与率", state.builderConfig.participation_rate, 0.01)}
          ${renderNumberField("rounds", "训练轮数", state.builderConfig.rounds, 1)}
          ${renderNumberField("local_epochs", "本地 Epoch", state.builderConfig.local_epochs, 1)}
          ${renderNumberField("batch_size", "批大小", state.builderConfig.batch_size, 1)}
          ${renderNumberField("learning_rate", "学习率", state.builderConfig.learning_rate, 0.001)}
          ${renderNumberField("seed", "随机种子", state.builderConfig.seed, 1)}
          ${showTrustParams ? renderNumberField("trust_threshold", "可信筛选阈值", state.builderConfig.trust_threshold, 0.01) : ""}
          ${showDistillParams ? renderNumberField("distill_weight", "本地蒸馏权重", state.builderConfig.distill_weight, 0.01) : ""}
          ${showDistillParams ? renderNumberField("global_distill_weight", "全局蒸馏权重", state.builderConfig.global_distill_weight, 0.01) : ""}
          ${showDistillParams ? renderNumberField("temperature", "蒸馏温度", state.builderConfig.temperature, 0.1) : ""}
          <div class="form-field full">
            <label>实验模板</label>
            <div class="actions">
              ${templates
                .map(
                  (template) => `
                    <button type="button" class="secondary-button" data-template-id="${template.id}">
                      ${escapeHtml(template.name)}
                    </button>
                  `
                )
                .join("")}
            </div>
          </div>
          <div class="form-field full">
            <div class="actions">
              <button type="submit" class="primary-button">创建并进行运行预检</button>
            </div>
          </div>
        </form>
      </div>

      <div class="panel builder-preview-panel">
        <h3>配置预览</h3>
        <p class="muted">实时查看当前实验配置。</p>
        <pre class="config-preview" id="config-preview">${escapeHtml(JSON.stringify(state.builderConfig, null, 2))}</pre>
      </div>
    </section>
  `;

  const form = rootEl.querySelector("#builder-form");
  const assistantForm = rootEl.querySelector("#config-assistant-form");
  const assistantMessage = rootEl.querySelector("#config-assistant-message");
  rootEl.querySelectorAll("[data-assistant-provider]").forEach((button) => {
    button.addEventListener("click", () => {
      state.configAssistant.llm = state.configAssistant.llm || {
        provider: "local",
        baseUrl: "",
        apiKey: "",
        model: "",
      };
      const provider = button.dataset.assistantProvider === "api" ? "api" : "local";
      state.configAssistant.llm.provider = provider;
      if (provider === "local") {
        state.configAssistant.llm.baseUrl = "";
        state.configAssistant.llm.model = "";
      }
      state.configAssistant.error = null;
      renderBuilder();
    });
  });
  rootEl.querySelectorAll("[data-assistant-llm-field]").forEach((input) => {
    input.addEventListener("input", (event) => {
      state.configAssistant.llm = state.configAssistant.llm || {
        provider: "local",
        baseUrl: "",
        apiKey: "",
        model: "",
      };
      const field = event.target.dataset.assistantLlmField;
      if (field) {
        state.configAssistant.llm[field] = event.target.value;
      }
      state.configAssistant.error = null;
    });
  });
  assistantMessage?.addEventListener("input", (event) => {
    state.configAssistant.message = event.target.value;
    state.configAssistant.error = null;
  });
  rootEl.querySelectorAll("[data-assistant-example]").forEach((button) => {
    button.addEventListener("click", () => {
      state.configAssistant.message = button.dataset.assistantExample || "";
      state.configAssistant.error = null;
      if (assistantMessage) {
        assistantMessage.value = state.configAssistant.message;
        assistantMessage.focus();
      }
    });
  });
  assistantForm?.addEventListener("submit", async (event) => {
    event.preventDefault();
    const message = state.configAssistant.message.trim();
    if (!message) {
      state.configAssistant.error = "请先描述你想要的实验。";
      renderBuilder();
      return;
    }
    const llm = state.configAssistant.llm || {};
    const provider = llm.provider === "api" ? "api" : "local";
    const baseUrl = String(llm.baseUrl || "").trim();
    const model = String(llm.model || "").trim();
    const apiKey = String(llm.apiKey || "").trim();
    if (provider === "api") {
      if (!baseUrl) {
        state.configAssistant.error = "请先填写 API Base URL。";
        renderBuilder();
        return;
      }
      if (!model) {
        state.configAssistant.error = "请先填写模型名称。";
        renderBuilder();
        return;
      }
      if (!apiKey) {
        state.configAssistant.error = "调用 API 模型时请先填写 API Key。";
        renderBuilder();
        return;
      }
    }
    state.configAssistant.loading = true;
    state.configAssistant.error = null;
    state.configAssistant.draft = null;
    await renderBuilder();
    try {
      const draft = await api.parseExperimentConfig({
        message,
        current_config: state.builderConfig,
        module_id: state.builderConfig.module_id,
        llm: {
          provider,
          base_url: provider === "api" ? baseUrl : "",
          api_key: provider === "api" ? apiKey : "",
          model: provider === "api" ? model : "",
        },
      });
      state.configAssistant.draft = normalizeAssistantDraftLanguage(draft);
    } catch (error) {
      state.configAssistant.error = error.message || "实验场景智能体暂时不可用，请稍后重试。";
    } finally {
      state.configAssistant.loading = false;
      await renderBuilder();
    }
  });
  rootEl.querySelector("#clear-assistant-draft")?.addEventListener("click", () => {
    state.configAssistant.draft = null;
    state.configAssistant.error = null;
    renderBuilder();
  });
  rootEl.querySelector("#apply-assistant-config")?.addEventListener("click", async () => {
    const draft = state.configAssistant.draft;
    const draftConfig = draft?.config;
    if (!draftConfig) return;
    const missingFields = new Set(draft.missing_fields || []);
    const recommendedFields = new Set(draft.recommended_fields || []);
    const recognizedConfig = Object.fromEntries(
      Object.entries(draftConfig).filter(([field]) => (
        ASSISTANT_VISIBLE_FIELDS.has(field)
        && (!recommendedFields.size || recommendedFields.has(field))
        && !missingFields.has(field)
      ))
    );
    state.builderConfig = { ...state.builderConfig, ...recognizedConfig };
    state.configAssistant.draft = null;
    state.configAssistant.error = null;
    await renderBuilder();
    showAlert("success", "配置草案已应用到实验表单，请检查参数后再启动实验。");
  });
  form.addEventListener("input", (event) => {
    const target = event.target;
    if (!(target instanceof HTMLInputElement || target instanceof HTMLSelectElement)) {
      return;
    }
    const key = target.name;
    const rawValue = target.value;
    const hadAssistantDraft = Boolean(state.configAssistant.draft);
    const numericFields = new Set([
      "num_clients",
      "participation_rate",
      "silence_rate",
      "rounds",
      "local_epochs",
      "batch_size",
      "learning_rate",
      "seed",
      "trust_threshold",
      "distill_weight",
      "global_distill_weight",
      "temperature",
    ]);
    state.builderConfig[key] = numericFields.has(key) ? Number(rawValue) : rawValue;
    if (hadAssistantDraft) {
      state.configAssistant.draft = null;
      state.configAssistant.error = "检测到手动修改，旧配置草案已清除；如需继续使用自然语言，请重新生成。";
      renderBuilder();
      return;
    }
    if (["module_id", "task_type", "dataset", "defense"].includes(key)) {
      renderBuilder();
      return;
    }
    syncConfigPreview();
  });

  rootEl.querySelectorAll("[data-template-id]").forEach((button) => {
    button.addEventListener("click", async () => {
      const template = templates.find((item) => item.id === button.dataset.templateId);
      if (!template) {
        return;
      }
      const config = {
        ...state.builderConfig,
        ...template.config,
        num_clients: 30,
        rounds: 10,
      };
      if (!template.config.module_id && template.config.task_type && template.config.task_type !== state.builderConfig.task_type) {
        const taskModules = { vision_classification: "vision_benchmark", text_classification: "participation_adaptation", defense_demo: "robust_aggregation", reid: "reid_generalization", ulip3d: "multimodal_3d" };
        if (taskModules[template.config.task_type]) {
          config.module_id = taskModules[template.config.task_type];
        }
      }
      state.builderConfig = config;
      await renderBuilder();
      showAlert("info", `已载入模板：${template.name}`);
    });
  });

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const submitButton = form.querySelector('button[type="submit"]');
    if (!submitButton || submitButton.disabled) {
      return;
    }

    const defaultLabel = submitButton.textContent;
    submitButton.disabled = true;
    submitButton.textContent = "创建实验中…";
    showAlert("info", "正在创建实验并运行预检，本地 Qwen 检查可能需要几十秒，请不要重复点击。");

    try {
      const created = await api.createExperiment(state.builderConfig);
      state.selectedExperimentId = created.id;
      monitorBootExperimentId = created.id;
      monitorBootStage = "preflight";
      state.route = "monitor";
      if (location.hash.replace("#", "") !== "monitor") {
        location.hash = "monitor";
      }
      renderNav();
      submitButton.textContent = "运行预检中…";
      showAlert("info", "实验已创建，运行保障和安全监控智能体正在后台执行预检。");
      renderMonitorBoot(created.id, "preflight");

      // The two local-Qwen inspections are intentionally synchronous on the
      // backend, but the page should not remain on the builder while they
      // run.  Start the request in the background and let the monitor poll
      // the experiment status until it becomes ready_to_start or blocked.
      const preflightPromise = api.runPreflight(created.id);
      await renderMonitor();
      preflightPromise
        .then(async (workflow) => {
          if (state.route !== "monitor" || state.selectedExperimentId !== created.id) {
            return;
          }
          const blocked = workflow?.experiment_status === "blocked";
          showAlert(
            blocked ? "error" : "info",
            blocked
              ? "运行预检发现阻断项，请查看运行保障和安全监控报告。"
              : "配置草案已创建，运行预检完成，请确认后启动正式实验。"
          );
          await renderMonitor();
        })
        .catch(async (error) => {
          if (state.route === "monitor" && state.selectedExperimentId === created.id) {
            showAlert("error", `运行预检失败：${error?.message || "接口未返回成功状态"}`);
            await renderMonitor();
          }
        });
    } catch (error) {
      showAlert("error", `创建实验或运行预检失败：${error?.message || "接口未返回成功状态"}`);
    } finally {
      submitButton.disabled = false;
      submitButton.textContent = defaultLabel;
    }
  });
}

function syncConfigPreview() {
  const preview = rootEl.querySelector("#config-preview");
  if (preview) {
    preview.textContent = JSON.stringify(state.builderConfig, null, 2);
  }
}

const RUNTIME_AGENT_ORDER = [
  { name: "运行保障智能体", stage: "operations", label: "资源与复现" },
  { name: "安全监控智能体", stage: "security", label: "Agent 安全" },
];

function workflowStatusText(status) {
  return {
    completed: "已完成",
    passed: "已通过",
    normal: "正常",
    warning: "需关注",
    high_risk: "高风险",
    critical: "严重风险",
    running: "执行中",
    pending: "待执行",
    queued: "排队中",
    failed: "失败",
    blocked: "已阻断",
    stop_requested: "已请求停止",
    ready_to_start: "等待启动确认",
    analyzed: "分析完成",
  }[status] || status || "待执行";
}

function workflowActionText(action) {
  return {
    continue: "继续运行",
    pause_review: "暂停并人工复核",
    stop_and_review: "停止并复核",
  }[action] || "待确定";
}

function workflowProviderText(provider) {
  return {
    local: "本地模型",
    local_fallback: "本地回退",
    api: "API 模型",
    llm: "API 模型",
    llm_local: "本地模型",
    llm_api: "API 模型",
  }[provider] || provider || "未记录";
}

function localizedAgentText(value, agentName, status, disposition = "") {
  const text = String(value || "").trim();
  const localizedText = text
    .replace(/\bruntime\b/gi, "运行中周期检查")
    .replace(/\bpreflight_baseline\b/gi, "启动前安全基线")
    .replace(/\bpreflight\b/gi, "运行预检")
    .replace(/\bfinalize\b/gi, "完整性检查");
  if (/[\u3400-\u9fff]/.test(localizedText)) {
    return localizedText;
  }
  const lowerText = localizedText.toLowerCase();
  if (disposition === "pass" || lowerText === "pass") {
    return "该检查项已通过。";
  }
  if (lowerText.includes("evidence") || lowerText.includes("not_observed") || lowerText.includes("missing")) {
    return "当前检查所需证据不足，无法完成该项判断。";
  }
  if (agentName === "安全监控智能体") {
    return {
      normal: "安全检查已完成，当前未发现需要升级处理的安全观察。",
      warning: "部分安全检查需要人工复核。",
      high_risk: "安全检查发现高风险观察，建议停止并复核。",
      critical: "安全监控检查失败或发现严重风险，需停止并复核。",
    }[status] || "安全监控检查已完成，请查看检查结果。";
  }
  return {
    passed: "运行保障检查已完成，当前未发现需要升级处理的问题。",
    warning: "部分运行保障检查需要人工复核。",
    blocked: "运行保障检查发现阻断项，实验需停止并复核。",
  }[status] || "运行保障检查已完成，请查看检查结果。";
}

function normalizeSecurityDisplayReport(report) {
  if (!report) {
    return report;
  }
  const findings = [
    ...(report.agent_security_findings || []),
    ...(report.fl_security_findings || []),
  ];
  const insufficient = Boolean(report.missing_evidence?.length) || findings.some((finding) => (
    finding?.status === "not_observed" || /证据不足|evidence|missing/i.test(String(finding?.message || ""))
  ));
  if (!insufficient) {
    return report;
  }
  return {
    ...report,
    status: report.status === "normal" ? "warning" : report.status,
    recommended_action: report.recommended_action === "continue" ? "pause_review" : report.recommended_action,
  };
}

const SECURITY_EVIDENCE_FALLBACKS = {
  "evidence.agent_trace": {
    evidence_name: "智能体调用轨迹",
    what_is_missing: "工具名称、调用阶段、调用状态、错误、模型元数据和输出摘要等逐次轨迹。",
    why_it_matters: "无法逐次核对实际调用是否只使用允许的只读工具，也无法确认任务状态和输出链路是否完整。",
    possible_consequence: "可能漏检工具越权、未授权操作、异常调用链或任务状态伪造；这不表示这些问题已经发生。",
  },
  "evidence.allowed_tools": {
    evidence_name: "允许的工具边界",
    what_is_missing: "本次安全监控智能体实际可调用的工具白名单及能力说明。",
    why_it_matters: "无法将实际工具调用与平台批准的只读能力逐项比对。",
    possible_consequence: "可能漏检越权工具或执行型工具被调用的风险；不能据此证明工具权限安全。",
  },
  "evidence.allowed_tool_boundaries": {
    evidence_name: "各智能体的允许工具边界",
    what_is_missing: "按智能体名称列出的实际只读工具白名单，以及安全监控智能体自身的有效能力边界。",
    why_it_matters: "无法将每个智能体的调用轨迹与其对应的批准工具集合逐项比对。",
    possible_consequence: "可能把合法工具误判为越权，也可能漏检某个智能体调用了未授权工具；不能据此证明工具权限安全。",
  },
  "evidence.experiment.config": {
    evidence_name: "批准的实验配置",
    what_is_missing: "任务身份、数据集、算法、攻击/防御、客户端和运行参数等批准配置。",
    why_it_matters: "无法确认运行期间的任务上下文是否保持不变。",
    possible_consequence: "可能漏检配置漂移、任务身份替换或偏离批准范围的运行行为。",
  },
  "evidence.experiment.config.num_clients": {
    evidence_name: "批准的客户端数量",
    what_is_missing: "实验批准的客户端总数。",
    why_it_matters: "无法将实际参与客户端数量与批准规模核对。",
    possible_consequence: "可能漏检客户端规模异常或参与范围偏离配置。",
  },
  "evidence.experiment.config.participation_rate": {
    evidence_name: "批准的客户端参与率",
    what_is_missing: "实验批准的每轮客户端参与率。",
    why_it_matters: "无法判断每轮参与数量是否符合实验协议。",
    possible_consequence: "可能漏检参与率异常、抽样逻辑偏移或参与协议改变。",
  },
  "evidence.experiment.config.attack": {
    evidence_name: "批准的攻击配置",
    what_is_missing: "本次实验是否配置攻击以及攻击类型。",
    why_it_matters: "无法区分异常指标是实验设计的一部分还是运行异常。",
    possible_consequence: "可能误读或漏读联邦学习运行观测；不能据此确认存在攻击。",
  },
  "evidence.experiment.config.defense": {
    evidence_name: "批准的防御配置",
    what_is_missing: "本次实验是否配置防御以及防御类型。",
    why_it_matters: "无法判断防御指标是否应该出现以及观测是否覆盖当前实验。",
    possible_consequence: "可能漏检防御失效或错误解释检测指标。",
  },
  "evidence.events": {
    evidence_name: "实验事件和工具输出摘要",
    what_is_missing: "运行阶段、轮次、错误、控制动作以及不可信事件文本等记录。",
    why_it_matters: "无法还原异常发生前后的调用顺序，也无法检查事件文本中的间接提示注入。",
    possible_consequence: "可能漏检异常控制请求、提示注入或运行状态伪造。",
  },
  "evidence.fl_round_snapshots": {
    evidence_name: "联邦学习轮次快照",
    what_is_missing: "每轮参与客户端、指标、梯度漂移、聚合和防御观测等原始状态。",
    why_it_matters: "无法把安全判断对应到具体轮次，也无法区分单轮异常和持续性风险。",
    possible_consequence: "可能漏检客户端参与异常、更新异常或梯度漂移风险。",
  },
  "evidence.metrics": {
    evidence_name: "实验指标序列",
    what_is_missing: "损失、精度、梯度漂移、客户端参与率和防御检测等按轮次指标。",
    why_it_matters: "无法检查指标是否有限、连续、与轮次一致或出现异常变化。",
    possible_consequence: "可能漏检指标篡改、异常更新或训练过程被扰动的信号。",
  },
  "evidence.manifest": {
    evidence_name: "实验和运行环境清单",
    what_is_missing: "配置指纹、环境指纹、随机种子、数据来源、训练模式和插件解析结果。",
    why_it_matters: "无法确认当前运行环境和插件边界是否与批准实验一致。",
    possible_consequence: "可能漏检环境漂移、插件替换或结果不可复核的问题。",
  },
  "有效的模型检查结果": {
    evidence_name: "本轮安全监控模型检查结果",
    what_is_missing: "本轮没有获得可解析的 Qwen JSON 检查结果。",
    why_it_matters: "安全监控智能体没有完成对智能体自身安全证据的审查。",
    possible_consequence: "本轮安全状态无法判定，可能漏检安全问题，因此需要停止并等待复核。",
  },
};

function securityEvidenceFallback(path) {
  const normalized = String(path || "").trim();
  if (SECURITY_EVIDENCE_FALLBACKS[normalized]) {
    return {
      ...SECURITY_EVIDENCE_FALLBACKS[normalized],
      evidence_paths: normalized.startsWith("evidence.") ? [normalized] : [],
      check_ids: [],
      observed_state: normalized === "有效的模型检查结果"
        ? "本轮未获得可解析的 Qwen JSON 检查结果"
        : (normalized === "[truncated]" ? "模型返回的证据路径被截断" : "已报告缺失或不足"),
      interpretation: "这是可观测性缺口，不代表已经确认发生攻击。",
    };
  }
  if (normalized === "[truncated]") {
    return {
      evidence_name: "具体证据路径",
      evidence_paths: [],
      check_ids: [],
      observed_state: "模型返回的证据路径被截断或无法解析",
      what_is_missing: "本轮检查没有给出可定位到具体 JSON 字段的证据缺口。",
      why_it_matters: "无法确认究竟是哪一项安全检查缺少输入，也无法复核模型判断依据。",
      possible_consequence: "可能漏检工具越权、提示注入、任务完整性或联邦学习状态异常。",
      interpretation: "这是可观测性缺口，不代表已经确认发生攻击。",
    };
  }
  return {
    evidence_name: normalized || "安全检查证据",
    evidence_paths: normalized.startsWith("evidence.") ? [normalized] : [],
    check_ids: [],
    observed_state: "已报告缺失或不足",
    what_is_missing: "当前检查没有提供足够的可定位证据。",
    why_it_matters: "无法将安全判断对应到可核验的输入证据。",
    possible_consequence: "可能漏检相关安全问题；这不表示问题已经发生。",
    interpretation: "这是可观测性缺口，不代表已经确认发生攻击。",
  };
}

function buildSecurityEvidenceGapDetails(report, taskOutput = {}) {
  const sourceReports = [report || {}, taskOutput || {}];
  const explicit = sourceReports.flatMap((item) => (
    Array.isArray(item.evidence_gap_details) ? item.evidence_gap_details : []
  )).filter((item) => item && typeof item === "object");
  const missing = sourceReports.flatMap((item) => (
    Array.isArray(item.missing_evidence) ? item.missing_evidence : []
  ));
  const findings = sourceReports.flatMap((item) => [
    ...(item.agent_security_findings || []),
    ...(item.fl_security_findings || []),
    ...(item.findings || []),
  ]);
  const candidates = [
    ...missing,
    ...findings.flatMap((finding) => (
      finding?.status === "not_observed" ? (finding.evidence_paths || []) : []
    )),
  ].map((item) => String(item || "").trim()).filter(Boolean);
  const insufficient = candidates.length > 0 || findings.some((finding) => (
    finding?.status === "not_observed" || /证据不足|evidence|missing/i.test(String(finding?.message || ""))
  ));
  if (!explicit.length && !insufficient) {
    return [];
  }
  const details = [...explicit];
  const known = new Set(details.map((item) => `${item.evidence_name || ""}|${(item.evidence_paths || []).join(",")}`));
  for (const candidate of candidates) {
    const detail = securityEvidenceFallback(candidate);
    const key = `${detail.evidence_name}|${(detail.evidence_paths || []).join(",")}`;
    if (!known.has(key)) {
      details.push(detail);
      known.add(key);
    }
  }
  if (!details.length) {
    details.push(securityEvidenceFallback("安全检查证据"));
  }
  return details.slice(0, 12);
}

function workflowStageText(stage) {
  return {
    scenario_configuration: "场景配置",
    preflight: "运行预检",
    preflight_baseline: "启动前安全基线",
    runtime: "运行中周期检查",
    finalize: "完整性检查",
    result_analysis: "结果分析",
  }[stage] || stage || "工作流节点";
}

function renderAgentWorkflowPanel(workflowSnapshot, experiment) {
  const tasks = workflowSnapshot?.agents || [];
  const findTask = (agent) => {
    const matches = tasks.filter((task) => task.agent_name === agent.name);
    if (!matches.length) return null;
    return matches[matches.length - 1];
  };
  const operations = workflowSnapshot?.operations_report;
  const security = normalizeSecurityDisplayReport(workflowSnapshot?.security_report);
  const experimentStatus = workflowSnapshot?.experiment_status || experiment?.status || "draft";
  const runtimeInterval = Number(workflowSnapshot?.runtime_agent_check_interval_rounds || 3);
  const runtimeCheckCount = Number(workflowSnapshot?.runtime_check_count || 0);
  const canStart = experimentStatus === "ready_to_start";
  const operationFindings = operations?.findings || [];
  const securityFindings = [
    ...(security?.agent_security_findings || []),
    ...(security?.fl_security_findings || []),
  ];
  const agentCards = RUNTIME_AGENT_ORDER.map((agent) => {
    const task = findTask(agent);
    const report = agent.name === "运行保障智能体" ? operations : security;
    const status = experimentStatus === "running" && agent.name === "安全监控智能体"
      ? "running"
      : (report?.status || task?.status || "pending");
    const output = task?.output || {};
    const llmMetadata = output?.llm_metadata || {};
    const evidenceGapDetails = agent.name === "安全监控智能体"
      ? buildSecurityEvidenceGapDetails(security, output)
      : [];
    const inspectionSkill = output?.inspection_skill || llmMetadata.inspection_skill;
    const inspectionHash = output?.inspection_snapshot_hash || llmMetadata.inspection_snapshot_hash;
    const provider = task?.provider || llmMetadata.provider;
    const model = task?.model || llmMetadata.model;
    const usedLlm = llmMetadata.used_llm === true;
    const modelText = model || workflowProviderText(provider);
    const llmStateText = inspectionSkill
      ? (usedLlm ? "Qwen 已完成 JSON 检查" : "未得到有效 LLM 检查结果")
      : (model ? "模型调用已记录" : "该节点无检查快照");
    const snapshotText = output?.inspection_snapshot
      ? JSON.stringify(output.inspection_snapshot, null, 2)
      : "当前任务没有保存 inspection JSON。";
    const message = localizedAgentText(
      output?.summary?.message || output?.message || (status === "pending" ? "等待上游阶段完成。" : "任务已记录。"),
      agent.name,
      status,
    );
    const evidence = agent.name === "运行保障智能体"
      ? {
          status: operations?.status || status,
          message: operationFindings.length
            ? localizedAgentText(
                operationFindings[0].message,
                agent.name,
                operations?.status || status,
                operationFindings[0].status,
              )
            : (operations?.manifest?.data_source ? `数据来源：${operations.manifest.data_source}` : "等待预检报告"),
          meta: `复现指纹：${operations?.manifest?.reproduction_fingerprint ? String(operations.manifest.reproduction_fingerprint).slice(0, 16) : "未生成"}`,
        }
      : {
          status: security?.status || status,
          message: securityFindings.length
            ? localizedAgentText(
                securityFindings[0].message,
                agent.name,
                security?.status || status,
                securityFindings[0].status,
              )
            : (evidenceGapDetails.length
              ? `证据不足：${evidenceGapDetails[0].evidence_name}。${evidenceGapDetails[0].possible_consequence}`
              : (security ? "尚未发现需要升级处理的安全观察" : "等待安全监控报告")),
          meta: `安全观察 ${Number(security?.observation_count || 0)} 条，运行中周期检查 ${runtimeCheckCount} 次，建议动作：${workflowActionText(security?.recommended_action)}`,
        };
    return `
      <article class="agent-workflow-card workflow-status-${escapeClassName(status)}">
        <div class="agent-workflow-card-topline">
          <span class="workflow-status-dot" aria-hidden="true"></span>
          <span class="workflow-agent-stage">${escapeHtml(agent.label)}</span>
          <span class="workflow-status-label">${escapeHtml(workflowStatusText(status))}</span>
        </div>
        <h4>${escapeHtml(agent.name)}</h4>
        <p class="muted">${escapeHtml(workflowStageText(task?.stage || agent.stage))}</p>
        <p>${escapeHtml(message)}</p>
        <small class="workflow-tool-count">工具调用 ${Number(task?.tool_calls?.length || 0)} 次</small>
        <small class="workflow-llm-meta">${escapeHtml(llmStateText)} · ${escapeHtml(modelText)}</small>
        <div class="agent-workflow-card-evidence">
          <div class="agent-workflow-card-evidence-heading">
            <span>当前检查证据</span>
            <span class="workflow-inline-state">${escapeHtml(workflowStatusText(evidence.status))}</span>
          </div>
          <p>${escapeHtml(evidence.message)}</p>
          <small>${escapeHtml(evidence.meta)}</small>
        </div>
        ${agent.name === "安全监控智能体" && evidenceGapDetails.length ? `
          <details open class="workflow-inspection-details workflow-evidence-gap-details">
            <summary>查看证据不足说明（${evidenceGapDetails.length} 项）</summary>
            <ul>
              ${evidenceGapDetails.slice(0, 6).map((gap) => `
                <li>
                  <strong>${escapeHtml(gap.evidence_name || "未命名证据")}</strong>：
                  ${escapeHtml(gap.observed_state || "无法确认")}
                  ${gap.check_ids?.length ? `（涉及：${escapeHtml(gap.check_ids.join("、"))}）` : ""}。
                  ${escapeHtml(gap.what_is_missing || "缺少可核验内容。")}
                  可能导致：${escapeHtml(gap.possible_consequence || "无法排除相关风险。")}
                </li>
              `).join("")}
            </ul>
          </details>
        ` : ""}
        ${inspectionSkill ? `
          <details class="workflow-inspection-details">
            <summary>查看检查凭据</summary>
            <div class="workflow-inspection-facts">
              <span>skill：${escapeHtml(inspectionSkill)}</span>
              <span>快照 hash：${escapeHtml(inspectionHash ? String(inspectionHash).slice(0, 16) : "未记录")}</span>
            </div>
            <pre>${escapeHtml(snapshotText)}</pre>
          </details>
        ` : ""}
      </article>
    `;
  }).join("");

  return `
    <section class="panel compact-panel agent-workflow-panel" id="agent-workflow-panel">
      <div class="agent-workflow-header">
        <div>
          <p class="eyebrow">协同工作流</p>
          <h3>运行保障与安全监控共同守护实验</h3>
          <p class="muted">配置页先完成场景解析；运行中每 ${runtimeInterval} 轮由运行保障和安全监控共同检查，本页展示运行期证据和处置状态，实验结论请前往结果分析页查看。</p>
        </div>
        <div class="agent-workflow-header-actions">
          <span class="status-badge ${security?.status === "critical" ? "error" : security?.status === "high_risk" ? "warn" : "ok"}">${escapeHtml(workflowStatusText(experimentStatus))}</span>
          ${canStart ? '<button class="primary-button" id="confirm-start-experiment" type="button">确认并启动</button>' : ""}
        </div>
      </div>
      <div class="agent-workflow-grid">${agentCards}</div>
    </section>
  `;
}

function renderMonitorBoot(experimentId, stage = monitorBootStage) {
  const starting = stage === "starting";
  rootEl.innerHTML = `
    <section class="panel empty-state">
      <h3>${starting ? "实验启动中" : "运行预检中"}</h3>
      <p class="muted">${starting
        ? `实验 ${escapeHtml(experimentId)} 已通过预检，Runner 正在初始化并启动训练。`
        : `实验 ${escapeHtml(experimentId)} 已创建，运行保障和安全监控智能体正在执行预检。`
      }</p>
    </section>
  `;
}

async function renderMonitor() {
  if (monitorBootExperimentId && state.selectedExperimentId === monitorBootExperimentId) {
    renderMonitorBoot(monitorBootExperimentId, monitorBootStage);
  }
  let experimentId = state.selectedExperimentId;

  try {
    const exps = await api.getExperiments();
    const found = exps.find((e) => e.id === experimentId);
    if (!found && exps.length > 0) {
      experimentId = exps[0].id;
      state.selectedExperimentId = experimentId;
    } else if (exps.length === 0 && state.backendAvailable) {
      if (state.route !== "monitor") {
        return;
      }
      rootEl.innerHTML = `<div class="panel empty-state"><h3>暂无实验数据</h3><p class="muted">后端还没有实验记录，请先前往「实验配置」页创建并启动一个实验。</p></div>`;
      return;
    }
  } catch (_) {
    /* fetchWithFallback will already downgrade to mock data */
  }

  const [experiment, metrics, logs, workflowSnapshot] = await Promise.all([
    api.getExperiment(experimentId),
    api.getExperimentMetrics(experimentId),
    state.backendAvailable ? api.getExperimentLogs(experimentId, 200) : [],
    api.getWorkflow(experimentId),
  ]);
  if (state.route !== "monitor") {
    return;
  }
  if (monitorBootExperimentId === experiment.id) {
    const waitingForPreflight = ["draft", "preflight_pending"].includes(experiment.status);
    const hasBootstrappedMetrics =
      (experiment.current_round || 0) > 0 ||
      (metrics.history || []).length > 0 ||
      !waitingForPreflight;
    if (!hasBootstrappedMetrics) {
      renderMonitorBoot(experiment.id, monitorBootStage);
      window.setTimeout(() => {
        if (state.route === "monitor" && state.selectedExperimentId === experiment.id) {
          renderMonitor();
        }
      }, 1200);
      return;
    }
    monitorBootExperimentId = null;
    monitorBootStage = "preflight";
  }

  if (lastMonitorExperimentId !== experiment.id) {
    revealedRounds = Math.max(0, Number(experiment.current_round || 0));
    lastMonitorExperimentId = experiment.id;
    selectedHeatmapRoundValue = null;
  }
  const isRealtimeBackend = state.backendAvailable;
  if (isRealtimeBackend) {
    revealedRounds = Math.max(
      experiment.current_round || 0,
      metrics.history?.[metrics.history.length - 1]?.round || 0,
      revealedRounds
    );
  } else if (revealedRounds < (experiment.rounds || 100)) {
    revealedRounds = Math.min(revealedRounds + 1, experiment.rounds || 100);
  }
  if (!isRealtimeBackend) {
    metrics.history = (metrics.history || []).filter((h) => h.round <= revealedRounds);
    metrics.participation = (metrics.participation || []).filter((p) => p.round <= revealedRounds);
    metrics.defense = (metrics.defense || []).filter((d) => d.round <= revealedRounds);
  }
  metrics.participation = buildParticipationTimeline(metrics.participation || [], metrics.history || [], experiment);
  state.currentExperiment = experiment;
  const availableMonitorMetrics = getAvailableMonitorMetrics(experiment);
  if (!availableMonitorMetrics.some((metric) => metric.key === state.monitorMetric)) {
    state.monitorMetric = availableMonitorMetrics[0]?.key || "train_loss";
  }

  const logLines = [
    `[${new Date().toLocaleTimeString("zh-CN")}] 监听 EventSource('/api/experiments/${experimentId}/events/stream')`,
    `[${new Date().toLocaleTimeString("zh-CN")}] 当前状态: ${experiment.status}`,
    ...buildMonitorLogLines(logs || []),
  ];
  const shouldHoldHistoryAtBoot =
    state.backendAvailable &&
    monitorBootExperimentId === experiment.id &&
    experiment.status === "running" &&
    Number(experiment.current_round || 0) <= 1;
  const visibleHistory = shouldHoldHistoryAtBoot ? [] : (metrics.history || []);

  rootEl.innerHTML = `
    ${renderPageShowcase("monitor", "训练现场正在发光，每一轮都有证据", "用同一块监控屏追踪轮次进度、指标曲线、客户端参与和防御检测，让实验过程不再只是后台日志。", [
      { label: "实验状态", value: renderMonitorStatusText(experiment.status, experiment.current_round, experiment.rounds, experiment.created_at) },
      { label: "当前轮次", value: `${experiment.current_round || 0}/${experiment.rounds || 0}` },
      { label: "客户端", value: `${experiment.num_clients || 0} 个` },
      { label: "数据来源", value: renderMonitorDataSource(experiment, metrics) },
    ])}
    ${renderAgentWorkflowPanel(workflowSnapshot, experiment)}
    <section class="monitor-layout">
      <div class="panel compact-panel monitor-progress-panel">
        <h3>轮次进度</h3>
        <div class="list">
          <div class="list-item">
            <strong>实验 ID</strong>
            <p class="metric-value">${escapeHtml(experiment.id)}</p>
          </div>
          <div class="list-item">
            <strong>当前轮次 / 总轮次</strong>
            <p class="metric-value"><span id="current-round">${experiment.current_round}</span> / ${experiment.rounds}</p>
          </div>
          <div class="list-item">
            <strong>状态</strong>
            <p class="metric-value" id="monitor-status">${escapeHtml(renderMonitorStatusText(experiment.status, experiment.current_round, experiment.rounds, experiment.created_at))}</p>
          </div>
          <div class="list-item">
            <strong>数据来源</strong>
            <p class="metric-value">${escapeHtml(renderMonitorDataSource(experiment, metrics))}</p>
          </div>
        </div>
        <div class="actions monitor-progress-actions">
          <button class="secondary-button" id="stop-experiment-button" type="button" ${["draft", "ready_to_start", "blocked", "completed", "analyzed", "stopped", "failed"].includes(experiment.status) ? "disabled" : ""}>${experiment.status === "stopped" ? "已停止" : "停止实验"}</button>
        </div>
      </div>

      <div class="panel compact-panel monitor-heatmap-panel">
        <h3>客户端参与热力图</h3>
        <p class="muted">按通信轮次查看客户端参与、沉默与恶意标记，缺失事件会由历史指标补全。</p>
        <div class="heatmap-slider" id="heatmap-slider">
          <button class="heatmap-nav-button" id="heatmap-prev" type="button" title="上一轮">&lsaquo;</button>
          <input type="range" id="heatmap-round-input" class="heatmap-range" min="0" value="0" max="${Math.max(0, metrics.participation.length - 1)}" step="1" title="拖拽切换轮次" />
          <button class="heatmap-nav-button" id="heatmap-next" type="button" title="下一轮">&rsaquo;</button>
          <span class="heatmap-round-label" id="heatmap-round-label">第 ${metrics.participation.length > 0 ? metrics.participation[metrics.participation.length - 1].round : 0} 轮</span>
        </div>
        <div id="heatmap-region">${renderHeatmap(metrics.participation, experiment.num_clients || 24)}</div>
      </div>

      <div class="panel monitor-chart-panel">
        <div class="monitor-chart-header">
          <div>
            <h3>指标曲线</h3>
            <p class="muted">展示训练过程中的历史指标变化，并同步保留关键轮次记录。</p>
          </div>
          <div class="metric-switch" id="monitor-metric-switch">
            ${availableMonitorMetrics
              .map(
                (metric) => `<button class="metric-switch-button ${state.monitorMetric === metric.key ? "active" : ""}" data-metric="${escapeHtml(metric.key)}" type="button">${escapeHtml(metric.label)}</button>`
              )
              .join("")}
          </div>
        </div>
        <div id="metric-line-chart-region">${renderSingleMetricChart(visibleHistory, state.monitorMetric)}</div>
        ${renderMonitorMetricHint(experiment)}
      </div>
    </section>

    <section class="monitor-bottom-grid">
      <div class="panel compact-panel monitor-lower-panel">
        <h3>防御面板</h3>
        <div id="defense-region">${renderDefenseCards(metrics.defense, metrics.participation)}</div>
      </div>

      <div class="panel compact-panel monitor-lower-panel">
        <h3>成果覆盖</h3>
        <div id="coverage-region">${renderLabCoveragePanel(experiment, metrics)}</div>
      </div>
    </section>

    <section class="panel">
      <h3>日志面板</h3>
      <div class="log-console" id="log-console">${escapeHtml(logLines.join("\n"))}</div>
    </section>
  `;

  rootEl.querySelectorAll("#monitor-metric-switch [data-metric]").forEach((button) => {
    button.addEventListener("click", () => {
      state.monitorMetric = button.dataset.metric;
      rootEl.querySelectorAll("#monitor-metric-switch [data-metric]").forEach((b) => b.classList.remove("active"));
      button.classList.add("active");
      const chartRegion = rootEl.querySelector("#metric-line-chart-region");
      if (chartRegion && (metrics.history || []).length > 0) {
        chartRegion.innerHTML = renderSingleMetricChart(metrics.history, state.monitorMetric);
      }
    });
  });

  rootEl.querySelector("#stop-experiment-button").addEventListener("click", async (event) => {
    const button = event.currentTarget;
    button.disabled = true;
    button.textContent = "停止中...";
    clearMonitorResources();
    experiment.status = "stopped";
    const currentRoundEl = rootEl.querySelector("#current-round");
    const statusEl = rootEl.querySelector("#monitor-status");
    const currentRound = Number(currentRoundEl?.textContent || experiment.current_round || 0);
    experiment.current_round = currentRound;
    if (statusEl) {
      statusEl.textContent = "已停止";
    }
    try {
      await api.stopExperiment(experimentId);
      button.textContent = "已停止";
      showAlert("info", `实验 ${experimentId} 已停止，监控流已关闭。`);
    } catch (error) {
      button.disabled = false;
      button.textContent = "停止实验";
      showAlert("error", `停止实验失败：${error.message || "接口未返回成功状态"}`);
    }
  });

  rootEl.querySelector("#confirm-start-experiment")?.addEventListener("click", async (event) => {
    const button = event.currentTarget;
    button.disabled = true;
    button.textContent = "提交中...";
    monitorBootExperimentId = experimentId;
    monitorBootStage = "starting";
    renderMonitorBoot(experimentId, "starting");
    try {
      await api.startExperiment(experimentId);
      showAlert("info", `实验 ${experimentId} 已通过预检并开始运行。`);
      await renderMonitor();
    } catch (error) {
      monitorBootExperimentId = null;
      monitorBootStage = "preflight";
      button.disabled = false;
      button.textContent = "确认并启动";
      showAlert("error", `启动实验失败：${error.message || "接口未返回成功状态"}`);
      await renderMonitor();
    }
  });

  if (experiment.status === "running" || (!state.backendAvailable && !["ready_to_start", "draft", "blocked"].includes(experiment.status))) {
    attachMonitorEventStream(experiment, metrics);
  }
}

function attachMonitorEventStream(experiment, metrics) {
  const experimentId = experiment.id;
  const currentRoundEl = rootEl.querySelector("#current-round");
  const statusEl = rootEl.querySelector("#monitor-status");
  const heatmapRegion = rootEl.querySelector("#heatmap-region");
  const heatmapSlider = rootEl.querySelector("#heatmap-slider");
  const heatmapInput = rootEl.querySelector("#heatmap-round-input");
  const heatmapLabel = rootEl.querySelector("#heatmap-round-label");
  const heatmapPrev = rootEl.querySelector("#heatmap-prev");
  const heatmapNext = rootEl.querySelector("#heatmap-next");
  const experimentMeta = mockDb.experiments.find((item) => item.id === experimentId) || { num_clients: 24 };
  const experimentNumClients = experiment.num_clients || experimentMeta.num_clients || 24;
  const syncHeatmapSelection = (preferredRound = null) => {
    if (!heatmapInput || !heatmapLabel || !heatmapRegion || !metrics.participation.length) {
      return;
    }
    const latestIdx = Math.max(0, metrics.participation.length - 1);
    heatmapInput.max = String(latestIdx);
    let targetIdx = latestIdx;
    if (preferredRound != null) {
      const foundIdx = metrics.participation.findIndex((item) => Number(item.round) === Number(preferredRound));
      if (foundIdx >= 0) {
        targetIdx = foundIdx;
      }
    }
    heatmapInput.value = String(targetIdx);
    selectedHeatmapRoundValue = metrics.participation[targetIdx]?.round ?? null;
    heatmapLabel.textContent = `第 ${metrics.participation[targetIdx]?.round ?? 0} 轮`;
    heatmapRegion.innerHTML = renderHeatmap(metrics.participation, experimentNumClients, targetIdx);
  };

  if (heatmapInput && heatmapPrev && heatmapNext) {
    heatmapInput.addEventListener("input", () => {
      const idx = parseInt(heatmapInput.value, 10);
      if (idx >= 0 && idx < metrics.participation.length) {
        selectedHeatmapRoundValue = metrics.participation[idx].round;
        heatmapLabel.textContent = `第 ${metrics.participation[idx].round} 轮`;
        heatmapRegion.innerHTML = renderHeatmap(metrics.participation, experimentNumClients, idx);
      }
    });
    heatmapPrev.addEventListener("click", () => {
      const current = parseInt(heatmapInput.value, 10);
      if (current > 0) {
        heatmapInput.value = current - 1;
        heatmapInput.dispatchEvent(new Event("input"));
      }
    });
    heatmapNext.addEventListener("click", () => {
      const current = parseInt(heatmapInput.value, 10);
      const maxIdx = parseInt(heatmapInput.max, 10);
      if (current < maxIdx) {
        heatmapInput.value = current + 1;
        heatmapInput.dispatchEvent(new Event("input"));
      }
    });
    if (metrics.participation.length) {
      syncHeatmapSelection(selectedHeatmapRoundValue);
    }
  }
  const defenseRegion = rootEl.querySelector("#defense-region");
  const logConsole = rootEl.querySelector("#log-console");
  const coverageRegion = rootEl.querySelector("#coverage-region");
  const metricChartRegion = rootEl.querySelector("#metric-line-chart-region");
  const appendLog = (line) => {
    logConsole.textContent = `${logConsole.textContent}\n${line}`;
    logConsole.scrollTop = logConsole.scrollHeight;
  };
  const refreshCoveragePanel = () => {
    if (coverageRegion) {
      coverageRegion.innerHTML = renderLabCoveragePanel(experiment, metrics);
    }
  };

  const replaceHistory = (nextHistory) => {
    if (!Array.isArray(nextHistory) || !nextHistory.length) {
      return;
    }
    metrics.history = nextHistory;
    if (metricChartRegion) {
      metricChartRegion.innerHTML = renderSingleMetricChart(metrics.history, state.monitorMetric);
    }
  };

  const handleRound = (payload) => {
    monitorBootExperimentId = null;
    monitorBootStage = "preflight";
    const previousLength = metrics.participation.length;
    const previousLatestRound = previousLength ? metrics.participation[previousLength - 1].round : null;
    const wasFollowingLatest = selectedHeatmapRoundValue == null || Number(selectedHeatmapRoundValue) === Number(previousLatestRound);
    currentRoundEl.textContent = String(payload.round);
    experiment.current_round = payload.round;
    if (statusEl) {
      const totalRounds = experiment.rounds || experiment.total_rounds || experimentMeta.rounds || mockDb.experiments.find((item) => item.id === experimentId)?.rounds || payload.round;
      statusEl.textContent = renderMonitorStatusText("running", payload.round, totalRounds, experiment.created_at);
    }
    metrics.history = [
      ...(metrics.history || []),
      {
        round: payload.round,
        train_loss: payload.train_loss,
        test_accuracy: payload.test_accuracy,
        map: payload.map ?? 0,
        rank1: payload.rank1 ?? 0,
        detection_accuracy: payload.detection_accuracy ?? 0,
        communication_cost_mb: payload.communication_cost_mb ?? 0,
        gradient_drift: payload.gradient_drift ?? 0,
        timestamp: payload.timestamp,
      },
    ];
    metrics.participation = buildParticipationTimeline(metrics.participation || [], metrics.history || [], experiment);
    syncHeatmapSelection(wasFollowingLatest ? payload.round : selectedHeatmapRoundValue);
    refreshCoveragePanel();
    if (metricChartRegion) {
      metricChartRegion.innerHTML = renderSingleMetricChart(metrics.history, state.monitorMetric);
    }
    appendLog(`[round_complete] round=${payload.round} train_loss=${payload.train_loss} test_accuracy=${payload.test_accuracy}`);
  };

  const syncFromBackendSnapshot = async () => {
    try {
      const [latestExperiment, latestMetrics] = await Promise.all([
        api.getExperiment(experimentId),
        api.getExperimentMetrics(experimentId),
      ]);
      experiment.current_round = latestExperiment.current_round;
      experiment.status = latestExperiment.status;
      experiment.finished_at = latestExperiment.finished_at;
      if (currentRoundEl) {
        currentRoundEl.textContent = String(latestExperiment.current_round || 0);
      }
      if (statusEl) {
        statusEl.textContent = renderMonitorStatusText(
          latestExperiment.status,
          latestExperiment.current_round,
          latestExperiment.rounds || latestExperiment.total_rounds,
          latestExperiment.created_at
        );
      }
      replaceHistory(latestMetrics.history || []);
      metrics.participation = buildParticipationTimeline(latestMetrics.participation || [], latestMetrics.history || [], latestExperiment);
      metrics.defense = latestMetrics.defense || [];
      defenseRegion.innerHTML = renderDefenseCards(metrics.defense, metrics.participation);
      syncHeatmapSelection(selectedHeatmapRoundValue);
      refreshCoveragePanel();
    } catch (_) {
      /* keep current UI when a transient poll fails */
    }
  };

  const handleParticipation = (payload) => {
    const previousLength = metrics.participation.length;
    const previousLatestRound = previousLength ? metrics.participation[previousLength - 1].round : null;
    const wasFollowingLatest = selectedHeatmapRoundValue == null || Number(selectedHeatmapRoundValue) === Number(previousLatestRound);
    metrics.participation = [...(metrics.participation || []).filter((item) => item.round !== payload.round), payload]
      .sort((a, b) => Number(a.round || 0) - Number(b.round || 0));
    syncHeatmapSelection(wasFollowingLatest ? payload.round : selectedHeatmapRoundValue);
    refreshCoveragePanel();
    appendLog(
      `[client_participation] round=${payload.round} participating=${payload.participating.length}  malicious=${payload.malicious.length}`
    );
  };

  const handleDefense = (payload) => {
    metrics.defense = [...(metrics.defense || []).filter((item) => item.round !== payload.round), payload];
    defenseRegion.innerHTML = renderDefenseCards(metrics.defense, metrics.participation);
    refreshCoveragePanel();
    appendLog(`[defense_detection] round=${payload.round} detection_accuracy=${payload.detection_accuracy}`);
  };

  const refreshWorkflowPanel = async () => {
    try {
      const latestWorkflow = await api.getWorkflow(experimentId);
      const panel = rootEl.querySelector("#agent-workflow-panel");
      if (panel) {
        panel.outerHTML = renderAgentWorkflowPanel(latestWorkflow, experiment);
      }
    } catch (_) {
      /* A transient workflow refresh must not interrupt the metric stream. */
    }
  };

  const handleCompleted = (payload, finalStatus) => {
    statusEl.textContent = finalStatus === "completed" ? "已完成" : "已停止";
    appendLog(`[${finalStatus}] ${payload.message}`);
  };

  if (state.backendAvailable) {
    const streamUrl = `${API_BASE}/api/experiments/${experimentId}/events/stream`;
    eventSourceRef = new EventSource(streamUrl);

    eventSourceRef.addEventListener("round_complete", (event) => handleRound(JSON.parse(event.data)));
    eventSourceRef.addEventListener("client_participation", (event) => handleParticipation(JSON.parse(event.data)));
    eventSourceRef.addEventListener("defense_detection", (event) => handleDefense(JSON.parse(event.data)));
    eventSourceRef.addEventListener("security_observation", (event) => {
      const payload = JSON.parse(event.data);
      appendLog(`安全监控：第 ${payload.round || "-"} 轮，状态：${workflowStatusText(payload.status)}，建议动作：${workflowActionText(payload.recommended_action)}。`);
      refreshWorkflowPanel();
    });
    eventSourceRef.addEventListener("runtime_agent_check", (event) => {
      const payload = JSON.parse(event.data);
      appendLog(`运行中周期检查：第 ${payload.round || "-"} 轮，状态：${workflowStatusText(payload.status)}，处置：${workflowActionText(payload.control_action)}。`);
      refreshWorkflowPanel();
    });
    eventSourceRef.addEventListener("agent_update", (event) => {
      const payload = JSON.parse(event.data);
      appendLog(`智能体更新：${payload.agent_name || "智能体"}，阶段：${workflowStageText(payload.stage)}，第 ${payload.round || "-"} 轮，状态：${workflowStatusText(payload.status)}。`);
      refreshWorkflowPanel();
    });
    eventSourceRef.addEventListener("result_analysis_completed", (event) => {
      const payload = JSON.parse(event.data);
      appendLog(`[result_analysis_completed] claim_level=${payload.claim_level}`);
      refreshWorkflowPanel();
    });
    eventSourceRef.addEventListener("completed", (event) => handleCompleted(JSON.parse(event.data), "completed"));
    eventSourceRef.addEventListener("stopped", (event) => handleCompleted(JSON.parse(event.data), "stopped"));
    eventSourceRef.addEventListener("done", () => {
      appendLog("[done] SSE 流已结束");
      eventSourceRef.close();
      eventSourceRef = null;
    });
    monitorTimer = window.setInterval(() => {
      syncFromBackendSnapshot();
    }, 4000);
    return;
  }

  let step = 0;
  const timeline = mockDb.eventTimelines[experimentId] || [];
  monitorTimer = window.setInterval(() => {
    const event = timeline[step];
    if (!event) {
      window.clearInterval(monitorTimer);
      monitorTimer = null;
      appendLog("[done] 演示事件回放结束");
      return;
    }
    if (event.type === "round_complete") {
      handleRound(event.payload);
    }
    if (event.type === "client_participation") {
      handleParticipation(event.payload);
    }
    if (event.type === "defense_detection") {
      handleDefense(event.payload);
    }
    if (event.type === "completed" || event.type === "stopped") {
      handleCompleted(event.payload, event.type);
    }
    step += 1;
  }, 1200);
}

async function renderResults() {
  const RESULTS_DISPLAY_LIMIT = 5;
  const [algorithms, datasets, defenses, attacks, recentResults, allResults] = await Promise.all([
    api.getAlgorithms(),
    api.getDatasets(),
    api.getDefenses(),
    api.getAttacks(),
    api.getRecentResults(10),
    api.getResults({ ...state.resultsFilter, limit: 50 }),
  ]);
  if (state.route !== "results") {
    return;
  }
  const filteredResults = (allResults || []).slice();
  const displayResults = filteredResults.slice(0, RESULTS_DISPLAY_LIMIT);
  const selectedResult = displayResults.find((item) => item.experiment_id === state.selectedExperimentId) || displayResults[0];
  const analysisExperimentId = selectedResult?.experiment_id || null;
  const resultWorkflow = analysisExperimentId
    ? await api.getWorkflow(analysisExperimentId)
    : null;
  const resultsSourceHint = renderResultsSourceHint(filteredResults, state.backendAvailable);
  const bestFinalAccuracy = displayResults.length
    ? Math.max(...displayResults.map((item) => Number(item.final_accuracy || 0)))
    : 0;

  rootEl.innerHTML = `
    ${renderPageShowcase("results", "结果不是表格，是一面实验论证墙", "训练曲线、筛选条件、消融视图和导出结果集中在一起，方便把系统能力讲成清楚的证据链。", [
      { label: "展示实验", value: `${displayResults.length}/${filteredResults.length}` },
      { label: "最高最终精度", value: formatPercent(bestFinalAccuracy) },
      { label: "当前指标", value: state.resultsMetric },
      { label: "数据模式", value: state.backendAvailable ? "真实后端" : "演示回退" },
    ])}
    <section class="panel compact-panel compare-chart-panel results-chart-panel">
      <div class="results-chart-topbar">
        <div>
          <h3>曲线对比</h3>
          <p class="muted">多实验训练曲线对比。当前的 robustness 指的是防御模块的检测精度，仅在带攻击/防御的实验里才有意义。</p>
        </div>
        <div class="metric-switch" id="results-metric-switch">
          <button class="metric-switch-button ${state.resultsMetric === "accuracy" ? "active" : ""}" data-metric="accuracy" type="button">Accuracy</button>
          <button class="metric-switch-button ${state.resultsMetric === "loss" ? "active" : ""}" data-metric="loss" type="button">Loss</button>
          <button class="metric-switch-button ${state.resultsMetric === "robustness" ? "active" : ""}" data-metric="robustness" type="button">Detection Accuracy</button>
        </div>
      </div>
      ${resultsSourceHint}
      ${renderResultAgentPanel(resultWorkflow, analysisExperimentId)}
      ${renderComparisonLineChart(displayResults)}
    </section>

    <section class="results-middle-layout">
      <div class="panel results-filter-panel">
        <h3>筛选条件</h3>
        <p class="muted">按任务、算法、攻击、防御、参与率和随机种子筛选实验结果，并支持导出。</p>
        <div class="form-grid">
          ${renderSelectField("results_algorithm", "算法", [["", "全部"], ...algorithms.map((item) => [item.id, item.name])], state.resultsFilter.algorithm || "")}
          ${renderSelectField("results_dataset", "数据集", [["", "全部"], ...datasets.map((item) => [item.id, item.name])], state.resultsFilter.dataset || "")}
          ${renderSelectField("results_defense", "防御策略", [["", "全部"], ...defenses.map((item) => [item.id, item.name])], state.resultsFilter.defense || "")}
          ${renderSelectField("results_attack", "攻击方式", [["", "全部"], ...attacks.map((item) => [item.id, item.name])], state.resultsFilter.attack || "")}
          ${renderSelectField("results_task_type", "任务类型", [
            ["", "全部"],
            ["vision_classification", "vision_classification"],
            ["reid", "reid"],
            ["ulip3d", "ulip3d"],
          ], state.resultsFilter.task_type || "")}
          ${renderNumberField("results_participation_rate", "参与率（留空=不限）", state.resultsFilter.participation_rate || "", 0.01)}
          ${renderNumberField("results_seed", "随机种子（留空=不限）", state.resultsFilter.seed || "", 1)}
        </div>
        <div class="actions" style="margin-top:16px;">
          <button id="filter-apply-button" class="primary-button" type="button">应用筛选</button>
        </div>
        <div class="actions" style="margin-top:8px;">
          <button id="export-csv-button" class="secondary-button" type="button">导出 CSV</button>
          <button id="export-markdown-button" class="secondary-button" type="button">导出 Markdown</button>
          <button id="export-json-button" class="secondary-button" type="button">导出 JSON</button>
        </div>
      </div>

      <div class="panel compact-panel results-ablation-panel">
        <h3>消融视图</h3>
        <p class="muted">与上方曲线、下方表格使用同一份筛选结果；当前页优先展示真实结果，只有后端不可用时才会退回模拟数据。</p>
        ${renderAblationTable(displayResults)}
      </div>
    </section>

    <section class="panel">
      <h3>对比表</h3>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>实验名称</th>
              <th>算法</th>
              <th>数据集</th>
              <th>攻击</th>
              <th>防御</th>
              <th>参与率</th>
              <th>Seed</th>
              <th>最佳精度</th>
              <th>最终精度</th>
              <th>mAP</th>
              <th>Rank-1</th>
              <th>检测精度</th>
              <th>最优损失</th>
              <th>轮次</th>
              <th>状态</th>
            </tr>
          </thead>
          <tbody>
            ${displayResults
              .map(
                (row) => `
                  <tr>
                    <td>${escapeHtml(row.experiment_name)}</td>
                    <td>${escapeHtml(row.algorithm || "-")}</td>
                    <td>${escapeHtml(row.dataset || "-")}</td>
                    <td>${escapeHtml(row.attack || "-")}</td>
                    <td>${escapeHtml(row.defense || "-")}</td>
                    <td>${row.participation_rate ?? "-"}</td>
                    <td>${row.seed ?? "-"}</td>
                    <td>${formatPercent(row.best_accuracy)}</td>
                    <td>${formatPercent(row.final_accuracy)}</td>
                    <td>${formatPercent(row.map ?? 0)}</td>
                    <td>${formatPercent(row.rank1 ?? 0)}</td>
                    <td>${formatDetectionAccuracy(row.detection_accuracy ?? 0, row)}</td>
                    <td>${(row.best_loss ?? 0).toFixed(3)}</td>
                    <td>${row.communication_rounds}</td>
                    <td>${escapeHtml(row.status)}</td>
                  </tr>
                `
              )
              .join("")}
          </tbody>
        </table>
      </div>
    </section>
  `;

  rootEl.querySelectorAll("#results-metric-switch [data-metric]").forEach((button) => {
    button.addEventListener("click", () => {
      state.resultsMetric = button.dataset.metric;
      renderResults();
    });
  });

  rootEl.querySelector("#filter-apply-button").addEventListener("click", async () => {
    const getVal = (sel) => rootEl.querySelector(sel)?.value || "";
    const query = {};
    const algo = getVal("#results_algorithm"); if (algo) query.algorithm = algo;
    const ds = getVal("#results_dataset"); if (ds) query.dataset = ds;
    const def = getVal("#results_defense"); if (def) query.defense = def;
    const atk = getVal("#results_attack"); if (atk) query.attack = atk;
    const tt = getVal("#results_task_type"); if (tt) query.task_type = tt;
    const pr = getVal("#results_participation_rate"); if (pr) query.participation_rate = Number(pr);
    const seed = getVal("#results_seed"); if (seed) query.seed = Number(seed);
    state.resultsFilter = query;
    showAlert("info", Object.keys(query).length > 0 ? `已应用 ${Object.keys(query).length} 个筛选条件` : "已清除筛选条件");
    await renderResults();
  });

  rootEl.querySelector("#export-csv-button").addEventListener("click", async () => {
    const result = await api.exportResults("csv", displayResults.map((item) => item.experiment_id));
    showAlert("info", `已触发导出：${result.message}`);
  });

  rootEl.querySelector("#export-markdown-button").addEventListener("click", async () => {
    const result = await api.exportResults("markdown");
    showAlert("info", `已触发导出：${result.message}`);
  });

  rootEl.querySelector("#export-json-button").addEventListener("click", async () => {
    const result = await api.exportResults("json");
    showAlert("info", `已触发导出：${result.message}`);
  });

  rootEl.querySelector("#regenerate-result-analysis")?.addEventListener("click", async (event) => {
    const button = event.currentTarget;
    button.disabled = true;
    button.textContent = "分析中...";
    try {
      await api.regenerateResultAnalysis(analysisExperimentId);
      showAlert("success", "结果分析已重新生成。定量统计仍由后端确定性代码提供。");
      await renderResults();
    } catch (error) {
      button.disabled = false;
      button.textContent = "重新分析";
      showAlert("error", `重新分析失败：${error.message || "接口未返回成功状态"}`);
    }
  });
}

async function renderApps() {
  const [overview, visionModule, reidModule, fed3dModule] = await Promise.all([
    api.getOverview(),
    api.getModuleById("vision_benchmark"),
    api.getModuleById("reid_generalization"),
    api.getModuleById("multimodal_3d"),
  ]);
  if (state.route !== "apps") {
    return;
  }

  const scenarios = [
    {
      id: "fedvision",
      eyebrow: "视觉分类应用演示",
      title: "FedVision",
      module: visionModule,
      summary: "面向图像分类成果展示：观众可以看到数据如何分布到不同客户端，算法如何在异构场景下提升稳定性，以及最终结果如何对比。",
      chips: ["视觉分类", "IID / Dirichlet / Pathological", "FedCVC / FedCADS / GFed-HSAM"],
      metrics: [
        { label: "场景数据集", value: "4", hint: "MNIST、CIFAR-10/100、Tiny-ImageNet" },
        { label: "划分协议", value: "3", hint: "IID、Dirichlet、Pathological" },
        { label: "算法方案", value: `${moduleAlgorithmCount(visionModule)} 个`, hint: "主方法与参考基线同台对比" },
      ],
      proof: [
        { label: "用户问题", text: "数据分散在不同客户端时，普通平均是否还能稳定提升视觉模型表现？" },
        { label: "展示价值", text: "用一套界面同时呈现数据划分、训练过程和算法对比，让观众直接看到联邦学习的作用。" },
        { label: "可见证据", text: "从样本图、客户端分布到结果曲线，形成从任务到结论的完整讲述链路。" },
      ],
      workflow: ["选择任务", "分配客户端", "联邦训练", "查看结果"],
      actions: [
        {
          label: "配置该场景实验",
          route: "builder",
          kind: "primary",
          config: {
            module_id: "vision_benchmark",
            task_type: "vision_classification",
            dataset: "cifar10",
            split: "dirichlet",
            protocol: "visual_classification",
            network: "resnet18",
            algorithm: "fedcads",
          },
        },
        { label: "查看视觉结果", route: "results", filter: { task_type: "vision_classification" } },
      ],
      cards: [
        {
          title: "任务样例",
          subtitle: "让观众先看见模型要学习的对象。",
          lines: ["CIFAR-10 / CIFAR-100：自然图像分类", "MNIST：轻量快速演示", "Tiny-ImageNet：更高类别数的视觉基准"],
          images: [
            { src: "samples/分类任务样例/cifar-10.png", alt: "CIFAR-10" },
            { src: "samples/分类任务样例/mnist.png", alt: "MNIST" },
          ],
        },
        {
          title: "异构划分",
          subtitle: "展示客户端数据分布从均衡到极端偏移。",
          lines: ["IID：类别比例近似一致", "Dirichlet(alpha=0.3)：长尾与类间失衡", "Pathological：单客户端只覆盖少量类别"],
          images: [
            { src: "samples/异构划分可视化/IID.png", alt: "IID split" },
            { src: "samples/异构划分可视化/Dirichlet.png", alt: "Dirichlet split" },
            { src: "samples/异构划分可视化/Pathological.png", alt: "Pathological split" },
          ],
          wide: true,
        },
        {
          title: "算法效果对比",
          subtitle: "同一任务下直接比较不同联邦方法的收益。",
          bars: [
            { label: "FedCADS", value: 0.858, hint: "双蒸馏与参与感知优化" },
            { label: "GFed-HSAM", value: 0.852, hint: "混合 sharpness-aware 优化" },
            { label: "FedCVC", value: 0.846, hint: "虚拟补偿缓解双重漂移" },
            { label: "FedAvg", value: 0.823, hint: "标准联邦平均基线" },
          ],
        },
      ],
    },
    {
      id: "fedreid",
      eyebrow: "域泛化 ReID",
      title: "FedReID",
      module: reidModule,
      summary: "面向行人重识别场景：不同摄像头、不同地点的数据无法集中时，系统展示如何获得更稳定的跨域识别能力。",
      chips: ["CO-EVO", "domain-as-client", "mAP / Rank-1"],
      metrics: [
        { label: "场景数据集", value: "4", hint: "CUHK02、CUHK03、MSMT17、Market1501" },
        { label: "评测协议", value: "3", hint: "leave-one-domain-out 等域泛化设置" },
        { label: "算法方案", value: `${moduleAlgorithmCount(reidModule)} 个`, hint: "ReID 与域泛化基线覆盖" },
      ],
      proof: [
        { label: "用户问题", text: "摄像头画质、光照和背景差异大时，模型能否识别同一个人？" },
        { label: "展示价值", text: "将跨域数据、身份语义和风格差异合在一个可解释场景里，说明联邦域泛化的必要性。" },
        { label: "可见证据", text: "mAP 和 Rank-1 指标直接给出识别质量，适合向非开发观众解释模型收益。" },
      ],
      workflow: ["选择数据域", "模拟摄像头客户端", "联邦泛化训练", "查看识别指标"],
      actions: [
        {
          label: "配置 ReID 实验",
          route: "builder",
          kind: "primary",
          config: {
            module_id: "reid_generalization",
            task_type: "reid",
            dataset: "market1501",
            split: "domain_as_client",
            protocol: "leave_one_domain_out",
            network: "tiny_reid",
            algorithm: "co_evo",
          },
        },
        { label: "查看 ReID 结果", route: "results", filter: { task_type: "reid" } },
      ],
      cards: [
        {
          title: "数据域差异",
          subtitle: "用跨摄像头图像说明数据分散和域差异。",
          lines: ["Market1501 / MSMT17：多摄像头场景", "CUHK02 / CUHK03：补充身份重识别协议", "每个域可作为客户端参与联邦训练"],
          images: [
            { src: "samples/ReID 数据域/Market-1501 Dataset.png", alt: "Market1501" },
            { src: "samples/ReID 数据域/MSMT17.png", alt: "MSMT17" },
          ],
        },
        {
          title: "泛化机制",
          subtitle: "把方法解释成用户能理解的识别稳定性来源。",
          lines: ["语义锚定：保持同一身份的核心特征", "风格库：吸收不同摄像头的外观偏移", "协同更新：同时提升身份表达和域适应能力"],
        },
        {
          title: "泛化指标摘要",
          subtitle: "mAP 与 Rank-1 同屏展示，适合答辩演示。",
          bars: [
            { label: "CO-EVO", value: 0.823, hint: "Rank-1 82.3%，mAP 78.1%" },
            { label: "FedReID", value: 0.795, hint: "Rank-1 79.5%，mAP 74.6%" },
            { label: "Client-only", value: 0.732, hint: "Rank-1 73.2%，mAP 69.8%" },
          ],
        },
      ],
    },
    {
      id: "fed3d",
      eyebrow: "3D 多模态联邦学习",
      title: "Fed3D",
      module: fed3dModule,
      summary: "面向 3D 感知场景：在不同机构拥有不同 3D 数据时，系统展示如何训练可迁移的点云表征并评估新类别泛化。",
      chips: ["FedULIP", "PointBERT", "base-to-new / cross-dataset / domain A-B-C-D"],
      metrics: [
        { label: "场景数据集", value: "6", hint: "ModelNet40、ScanObjectNN、ShapeNetCore 等" },
        { label: "评测协议", value: "3", hint: "base-to-new、cross-dataset、domain A/B/C/D" },
        { label: "算法方案", value: `${moduleAlgorithmCount(fed3dModule)} 个`, hint: "FedULIP 与 prompt-learning 基线" },
      ],
      proof: [
        { label: "用户问题", text: "3D 数据稀缺且分散时，模型能否迁移到没有见过的新类别？" },
        { label: "展示价值", text: "将点云样例、协议选择和泛化结果放在同一页面，突出 3D 联邦学习的应用潜力。" },
        { label: "可见证据", text: "base-to-new 与 cross-dataset 指标说明模型不是只记住训练类别，而是在学习可迁移表征。" },
      ],
      workflow: ["选择 3D 协议", "加载点云样本", "联邦训练", "评估泛化"],
      actions: [
        {
          label: "配置 FedULIP 实验",
          route: "builder",
          kind: "primary",
          config: {
            module_id: "multimodal_3d",
            task_type: "ulip3d",
            dataset: "modelnet40",
            split: "iid",
            protocol: "base2new",
            network: "pointbert",
            algorithm: "fedulip",
          },
        },
        { label: "查看 3D 结果", route: "results", filter: { task_type: "ulip3d" } },
      ],
      cards: [
        {
          title: "3D 输入形态",
          subtitle: "以点云样例展示 3D 联邦学习的任务对象。",
          lines: ["输入：Point Cloud + language prompt", "骨干：ULIP / PointBERT", "目标：跨客户端对齐 3D 语义表示"],
          image: "samples/点云.png",
        },
        {
          title: "评测协议",
          subtitle: "聚焦用户能理解的泛化能力检验。",
          lines: ["base-to-new：基类训练后评估新类别", "cross-dataset：跨数据集迁移测试", "domain A/B/C/D：多域 3D 场景泛化"],
        },
        {
          title: "泛化结果摘要",
          subtitle: "小批量演示仍保留完整评测协议结构。",
          bars: [
            { label: "base-to-new", value: 0.733, hint: "新类泛化主协议" },
            { label: "adapter + SACA", value: 0.705, hint: "适配器与语义校准" },
            { label: "cross-dataset", value: 0.684, hint: "跨数据集迁移测试" },
          ],
        },
      ],
    },
  ];

  rootEl.innerHTML = `
    ${renderAppsValueHero(scenarios, overview)}
    ${scenarios.map(renderAppScenarioPanel).join("")}
  `;

  bindRouteActions();
}

function moduleAlgorithmCount(module) {
  return (module?.algorithms || []).length;
}

function renderAppsValueHero(scenarios, overview) {
  const validatedCount = scenarios.reduce(
    (sum, scenario) => sum + (scenario.module?.algorithms || []).filter((algorithm) => algorithm.status === "validated").length,
    0
  );
  return `
    <section class="apps-hero">
      <div>
        <p class="eyebrow">Application Showcase</p>
        <h3>三类应用场景，展示联邦学习能解决什么问题</h3>
        <p class="apps-hero-lead">
          FedCompass 将视觉分类、行人重识别和 3D 感知组织成可观看的应用故事。
          用户先理解场景价值，再进入实验配置和结果对比，看到每个结论背后的训练证据。
        </p>
      </div>
      <div class="apps-hero-metrics">
        ${[
          { label: "应用场景", value: `${scenarios.length} 条`, hint: "视觉分类、ReID、3D 多模态" },
          { label: "已注册算法", value: `${overview.algorithm_count} 个`, hint: "与系统总览保持一致" },
          { label: "验证算法", value: `${validatedCount} 项`, hint: "三条场景内均可直接展示" },
          { label: "演示闭环", value: "4 步", hint: "场景、配置、监控、结果" },
        ]
          .map(
            (metric) => `
              <div class="apps-value-card">
                <span>${escapeHtml(metric.label)}</span>
                <strong>${escapeHtml(metric.value)}</strong>
                <p>${escapeHtml(metric.hint)}</p>
              </div>
            `
          )
          .join("")}
      </div>
    </section>
  `;
}

function renderAppScenarioPanel(scenario) {
  const module = scenario.module || {};
  return `
    <section class="app-scenario-section app-scenario-${escapeClassName(scenario.id)}">
      <div class="app-scenario-header">
        <div>
          <p class="eyebrow">${escapeHtml(scenario.eyebrow)}</p>
          <h3>${escapeHtml(scenario.title)}</h3>
          <p class="app-scenario-lead">${escapeHtml(scenario.summary)}</p>
          <div class="scenario-chip-row">
            ${(scenario.chips || []).map((chip) => `<span class="scenario-chip">${escapeHtml(chip)}</span>`).join("")}
          </div>
        </div>
        <div class="scenario-metric-grid">
          ${(scenario.metrics || []).map(renderScenarioMetric).join("")}
        </div>
      </div>

      <div class="scenario-value-grid">
        ${(scenario.proof || [])
          .map(
            (item) => `
              <div class="scenario-value-item">
                <span>${escapeHtml(item.label)}</span>
                <p>${escapeHtml(item.text)}</p>
              </div>
            `
          )
          .join("")}
      </div>

      <div class="scenario-workflow" aria-label="${escapeHtml(scenario.title)} 端到端流程">
        ${(scenario.workflow || [])
          .map(
            (step, index) => `
              <div class="scenario-workflow-step">
                <span class="scenario-step-index">${String(index + 1).padStart(2, "0")}</span>
                <strong>${escapeHtml(step)}</strong>
              </div>
            `
          )
          .join("")}
      </div>

      <div class="scenario-card-grid">
        ${(scenario.cards || []).map(renderAppScenarioCard).join("")}
      </div>

      <div class="app-scenario-actions">
        ${(scenario.actions || [])
          .map(
            (action) => `
              <button
                class="${action.kind === "primary" ? "primary-button" : "secondary-button"}"
                type="button"
                data-app-route="${escapeHtml(action.route)}"
                data-app-config="${escapeHtml(JSON.stringify(action.config || {}))}"
                data-app-filter="${escapeHtml(JSON.stringify(action.filter || {}))}"
              >
                ${escapeHtml(action.label)}
              </button>
            `
          )
          .join("")}
        <span class="scenario-service-tag">${escapeHtml(module.backend_service || "Module Service")}</span>
      </div>
    </section>
  `;
}

function renderScenarioMetric(metric) {
  return `
    <div class="scenario-mini-metric">
      <span>${escapeHtml(metric.label)}</span>
      <strong>${escapeHtml(metric.value)}</strong>
      <p>${escapeHtml(metric.hint)}</p>
    </div>
  `;
}

function renderAppScenarioCard(card) {
  return `
    <article class="app-evidence-card ${card.wide ? "wide" : ""}">
      <div class="app-evidence-card-head">
        <h4>${escapeHtml(card.title)}</h4>
        ${card.subtitle ? `<p>${escapeHtml(card.subtitle)}</p>` : ""}
      </div>
      ${card.lines ? renderScenarioLineList(card.lines) : ""}
      ${card.bars ? renderAppScenarioBars(card.bars) : ""}
      ${renderAppScenarioMedia(card)}
    </article>
  `;
}

function renderScenarioLineList(lines) {
  return `
    <ul class="scenario-line-list">
      ${lines.map((line) => `<li>${escapeHtml(line)}</li>`).join("")}
    </ul>
  `;
}

function renderAppScenarioMedia(card) {
  if (card.images) {
    return `
      <div class="app-card-images">
        ${card.images
          .map((img) => `<img src="${escapeHtml(img.src)}" alt="${escapeHtml(img.alt)}" title="${escapeHtml(img.alt)}" />`)
          .join("")}
      </div>
    `;
  }
  if (card.image) {
    return `<img src="${escapeHtml(card.image)}" alt="" class="app-card-single-img" />`;
  }
  return "";
}

function renderAppScenarioBars(bars) {
  return `
    <div class="scenario-bars">
      ${bars
        .map(
          (bar) => `
            <div class="scenario-bar-item">
              <div class="scenario-bar-meta">
                <span>${escapeHtml(bar.label)}</span>
                <span>${escapeHtml(bar.valueText || formatPercent(bar.value))}</span>
              </div>
              <div class="scenario-bar-track">
                <span class="scenario-bar-fill" style="width:${Math.max(bar.value * 100, 8)}%"></span>
              </div>
              <p class="muted">${escapeHtml(bar.hint)}</p>
            </div>
          `
        )
        .join("")}
    </div>
  `;
}

function bindRouteActions() {
  rootEl.querySelectorAll("[data-app-route]").forEach((button) => {
    button.addEventListener("click", () => {
      const route = button.dataset.appRoute;
      if (!routes.some((item) => item.id === route)) {
        return;
      }
      const config = JSON.parse(button.dataset.appConfig || "{}");
      const filter = JSON.parse(button.dataset.appFilter || "{}");
      if (route === "builder") {
        state.builderConfig = { ...state.builderConfig, ...config };
      }
      if (route === "results") {
        state.resultsFilter = filter;
      }
      navigateToRoute(route);
    });
  });
}

function renderMetricCard(title, value, hint) {
  return `
    <article class="panel metric-card">
      <p class="eyebrow">${escapeHtml(title)}</p>
      <div class="metric-value">${escapeHtml(String(value))}</div>
      <p class="muted">${escapeHtml(hint)}</p>
    </article>
  `;
}

function renderPageShowcase(kind, title, subtitle, metrics = []) {
  return `
    <section class="page-showcase page-showcase-${escapeClassName(kind)}">
      <div class="page-showcase-copy">
        <p class="eyebrow">FedCompass Showcase</p>
        <h3>${escapeHtml(title)}</h3>
        <p>${escapeHtml(subtitle)}</p>
      </div>
      <div class="page-showcase-visual" aria-hidden="true">
        <div class="page-showcase-rail">
          <span></span><span></span><span></span><span></span>
        </div>
        <div class="page-showcase-metrics">
          ${metrics
            .map(
              (metric) => `
                <div>
                  <span>${escapeHtml(metric.label)}</span>
                  <strong>${escapeHtml(metric.value)}</strong>
                </div>
              `
            )
            .join("")}
        </div>
      </div>
    </section>
  `;
}

function renderRunningExperiment(experiment) {
  return `
    <div class="list-item">
      <strong>${escapeHtml(experiment.name)}</strong>
      <p class="muted">${escapeHtml(experiment.algorithm)} · ${escapeHtml(experiment.dataset)}</p>
      <div class="pill-row">
        <span class="status-chip ${escapeClassName(experiment.status)}">${escapeHtml(experiment.status)}</span>
        <span class="pill">第 ${experiment.current_round}/${experiment.rounds} 轮</span>
        <span class="pill">Best ${formatPercent(experiment.best_accuracy)}</span>
      </div>
    </div>
  `;
}

function renderBarChart(items, valueKey, labelKey) {
  if (!items.length) {
    return renderEmptyState("当前接口还没有返回可展示的结果。");
  }
  return renderBarChartSvg(items, valueKey, labelKey, "实验名称", "指标值");
}

function renderSingleMetricChart(history, metricKey) {
  if (!history?.length) {
    return renderEmptyState("暂无可绘制的指标曲线。");
  }
  const labelMap = {
    train_loss: { label: "Train Loss", color: "#d86f45" },
    test_accuracy: { label: "Test Accuracy", color: "#0f6b62" },
    map: { label: "mAP", color: "#4f46e5" },
    rank1: { label: "Rank-1", color: "#7c3aed" },
    detection_accuracy: { label: "Detection Accuracy", color: "#b45309" },
  };
  const cfg = labelMap[metricKey] || labelMap.test_accuracy;
  const svg = renderMultiLineSvg({
    width: 860,
    height: 364,
    xValues: history.map((point) => point.round),
    xAxisLabel: "训练轮次",
    yAxisLabel: cfg.label,
    yPaddingRatio: metricKey === "train_loss" ? 0.18 : 0.08,
    series: [{ key: metricKey, label: cfg.label, color: cfg.color, values: history.map((point) => point[metricKey] ?? 0) }],
  });
  return `
    <div class="line-chart-header">
      <div class="line-chart-legend">
        <span class="legend-item"><i style="background:${cfg.color}"></i>${cfg.label}</span>
      </div>
    </div>
    <div class="line-chart-svg">${svg}</div>
  `;
}

function renderMetricLineChart(history, includeWrapper = true) {
  if (!history?.length) {
    return renderEmptyState("暂无可绘制的指标曲线。");
  }
  const svg = renderMultiLineSvg({
    width: 860,
    height: 364,
    xValues: history.map((point) => point.round),
    xAxisLabel: "训练轮次",
    yAxisLabel: "指标值",
    series: [
      { key: "train_loss", label: "Train Loss", color: "#d86f45", values: history.map((point) => point.train_loss) },
      { key: "test_accuracy", label: "Test Accuracy", color: "#0f6b62", values: history.map((point) => point.test_accuracy) },
      { key: "map", label: "mAP", color: "#4f46e5", values: history.map((point) => point.map ?? 0) },
      { key: "rank1", label: "Rank-1", color: "#7c3aed", values: history.map((point) => point.rank1 ?? 0) },
      { key: "detection_accuracy", label: "Detection Accuracy", color: "#b45309", values: history.map((point) => point.detection_accuracy ?? 0) },
    ],
  });
  const content = `
    <div class="line-chart-header">
      <div class="line-chart-legend">
        <span class="legend-item"><i style="background:#d86f45"></i>Train Loss</span>
        <span class="legend-item"><i style="background:#0f6b62"></i>Test Accuracy</span>
        <span class="legend-item"><i style="background:#4f46e5"></i>mAP</span>
        <span class="legend-item"><i style="background:#7c3aed"></i>Rank-1</span>
        <span class="legend-item"><i style="background:#b45309"></i>Detection Accuracy</span>
      </div>
    </div>
    <div class="line-chart-svg">${svg}</div>
  `;
  return includeWrapper ? `<div id="metric-line-chart-region">${content}</div>` : content;
}

function renderHeatmap(participation, totalClients = 24, selectedIdx = -1) {
  if (!participation.length) {
    return renderEmptyState("暂无客户端参与事件。");
  }
  const idx = selectedIdx >= 0 ? selectedIdx : participation.length - 1;
  const current = participation[idx];
  const participating = new Set((current.participating || []).map((item) => Number(item)));
  const malicious = new Set((current.malicious || []).map((item) => Number(item)));
  const heatCols = Math.max(6, Math.ceil(totalClients / 4));
  return `
    <div class="heatmap-wrap">
      <div class="heatmap-legend">
        <span class="legend-item"><i class="heat-cell"></i>未参与客户端</span>
        <span class="legend-item"><i class="heat-cell active"></i>参与客户端</span>
        <span class="legend-item"><i class="heat-cell malicious"></i>恶意标记</span>
      </div>
      <div class="heatmap" style="grid-template-columns:repeat(${heatCols},minmax(0,1fr))">
          ${Array.from({ length: totalClients }, (_, index) => {
            const clientId = index + 1;
            const classNames = ["heat-cell"];
            if (participating.has(clientId)) {
              classNames.push("active");
            }
            if (malicious.has(clientId)) {
              classNames.push("malicious");
            }
            return `<div class="${classNames.join(" ")}" title="第 ${current.round} 轮 · client_${clientId}"></div>`;
          }).join("")}
      </div>
    </div>
  `;
}

function buildParticipationTimeline(participationEvents = [], history = [], experiment = {}) {
  const config = experiment?.config_json || {};
  const totalClients = clampNumber(Number(experiment?.num_clients || config.num_clients || 30), 1, 500);
  const eventsByRound = new Map();
  (participationEvents || []).forEach((event) => {
    const round = Number(event?.round || 0);
    if (round > 0) {
      eventsByRound.set(round, normalizeParticipationEvent(event, totalClients));
    }
  });

  const historyByRound = new Map();
  (history || []).forEach((point) => {
    const round = Number(point?.round || 0);
    if (round > 0) {
      historyByRound.set(round, point);
    }
  });

  const rounds = Array.from(new Set([...historyByRound.keys(), ...eventsByRound.keys()]))
    .filter((round) => Number.isFinite(round) && round > 0)
    .sort((a, b) => a - b);

  return rounds.map((round) => (
    eventsByRound.get(round) || synthesizeParticipationEvent(round, totalClients, historyByRound.get(round), experiment)
  ));
}

function normalizeParticipationEvent(event, totalClients) {
  const cleanIds = (items) => Array.from(new Set((items || [])
    .map((item) => Number(item))
    .filter((item) => Number.isInteger(item) && item >= 1 && item <= totalClients)))
    .sort((a, b) => a - b);
  return {
    round: Number(event?.round || 0),
    participating: cleanIds(event?.participating),
    malicious: cleanIds(event?.malicious),
    synthetic: Boolean(event?.synthetic),
  };
}

function synthesizeParticipationEvent(round, totalClients, historyPoint, experiment = {}) {
  const config = experiment?.config_json || {};
  const baseRate = Number(historyPoint?.client_participation_rate ?? experiment?.participation_rate ?? config.participation_rate ?? 0.1);
  const safeRate = clampNumber(Number.isFinite(baseRate) ? baseRate : 0.1, 0.02, 1);
  const rng = createDeterministicRng(`${experiment?.id || "exp"}:${round}:participation`);
  const jitter = 0.84 + rng() * 0.32;
  const participatingCount = clampNumber(Math.round(totalClients * safeRate * jitter), 1, totalClients);
  const participating = sampleClientIds(totalClients, participatingCount, rng);
  const participatingSet = new Set(participating);
  const nonParticipants = Array.from({ length: totalClients }, (_, index) => index + 1)
    .filter((clientId) => !participatingSet.has(clientId));
  const hasAttack = (experiment?.attack || config.attack || "none") !== "none";
  const maliciousCount = hasAttack && round > 1 ? clampNumber(Math.round(totalClients * (0.03 + rng() * 0.04)), 1, Math.max(1, totalClients - participatingCount)) : 0;
  const maliciousPool = nonParticipants.length ? nonParticipants : Array.from({ length: totalClients }, (_, index) => index + 1);
  const malicious = sampleFromPool(maliciousPool, maliciousCount, rng);
  return {
    round,
    participating,
    malicious,
    synthetic: true,
  };
}

function sampleClientIds(totalClients, count, rng) {
  return sampleFromPool(Array.from({ length: totalClients }, (_, index) => index + 1), count, rng);
}

function sampleFromPool(pool, count, rng) {
  const items = pool.slice();
  for (let i = items.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [items[i], items[j]] = [items[j], items[i]];
  }
  return items.slice(0, count).sort((a, b) => a - b);
}

function createDeterministicRng(seedText) {
  let seed = 2166136261;
  for (const ch of String(seedText)) {
    seed ^= ch.charCodeAt(0);
    seed = Math.imul(seed, 16777619);
  }
  return () => {
    seed += 0x6d2b79f5;
    let t = seed;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function clampNumber(value, min, max) {
  return Math.min(max, Math.max(min, value));
}


function computeDrift(history) {
  if (!history || history.length < 2) {
    const fallback = Number(history?.[history.length - 1]?.gradient_drift || 0);
    return {
      ema_heterogeneity: 0,
      gradient_drift: Math.round(fallback * 1000) / 1000,
      compensation: Math.round(fallback * 0.62 * 1000) / 1000,
    };
  }
  const last5 = history.slice(-5);
  let ema = 0;
  for (let i = 1; i < last5.length; i++) {
    ema += Math.abs((last5[i].test_accuracy || 0) - (last5[i-1].test_accuracy || 0));
  }
  ema = Math.round((ema / Math.max(last5.length - 1, 1)) * 1000) / 1000;
  const grad = Math.round(Number(last5[last5.length - 1].gradient_drift || 0) * 1000) / 1000;
  const comp = Math.round(grad * 0.62 * 1000) / 1000;
  return { ema_heterogeneity: ema, gradient_drift: grad, compensation: comp };
}

function resolveMonitorDrift(metrics, preferBackendDrift = false) {
  const computedDrift = computeDrift(metrics?.history);
  if (preferBackendDrift && metrics?.drift) {
    return {
      ema_heterogeneity: formatMonitorDriftValue(metrics.drift.ema_heterogeneity),
      gradient_drift: formatMonitorDriftValue(metrics.drift.gradient_drift),
      compensation: formatMonitorDriftValue(metrics.drift.compensation),
    };
  }
  return {
    ema_heterogeneity: formatMonitorDriftValue(computedDrift.ema_heterogeneity),
    gradient_drift: formatMonitorDriftValue(computedDrift.gradient_drift),
    compensation: formatMonitorDriftValue(computedDrift.compensation),
  };
}

function formatMonitorDriftValue(value) {
  const num = Number(value || 0);
  return Number.isFinite(num) ? num.toFixed(4) : "0.0000";
}

function renderLabCoveragePanel(experiment, metrics) {
  const coverage = computeLabCoverage(experiment, metrics);
  return `
    <div class="coverage-grid">
      ${coverage.cards.map((card) => `
        <div class="coverage-card">
          <span class="coverage-label">${escapeHtml(card.label)}</span>
          <strong>${escapeHtml(card.value)}</strong>
          <small>${escapeHtml(card.detail)}</small>
        </div>
      `).join("")}
    </div>
    <div class="coverage-strip">
      ${coverage.tags.map((tag) => `<span class="pill">${escapeHtml(tag)}</span>`).join("")}
    </div>
  `;
}

function computeLabCoverage(experiment = {}, metrics = {}) {
  const config = experiment.config_json || {};
  const algorithm = experiment.algorithm || config.algorithm || "fedavg";
  const taskType = experiment.task_type || config.task_type || "vision_classification";
  const dataset = experiment.dataset || config.dataset || "cifar10";
  const defense = experiment.defense || config.defense || "none";
  const attack = experiment.attack || config.attack || "none";
  const history = metrics.history || [];
  const participation = metrics.participation || [];
  const defenseEvents = metrics.defense || [];
  const latest = history[history.length - 1] || {};
  const finalScore = resolvePrimaryScore(taskType, latest);
  const dataSource = renderMonitorDataSource(experiment, metrics);
  const defenseDetail = defense !== "none" || attack !== "none"
    ? `${formatAlgorithmDisplay(defense)} / ${formatAlgorithmDisplay(attack)}`
    : "未启用攻防链路";

  const tags = [
    getAlgorithmFamilyLabel(algorithm),
    getTaskFamilyLabel(taskType),
    dataSource.includes("真实数据") ? "真实数据联调" : "模拟快速演示",
    defense !== "none" || attack !== "none" ? "鲁棒安全链路" : "基础训练链路",
  ];
  if (participation.some((item) => item.synthetic)) {
    tags.push("轮次状态补全");
  }

  return {
    cards: [
      {
        label: "算法成果",
        value: formatAlgorithmDisplay(algorithm),
        detail: getAlgorithmFamilyLabel(algorithm),
      },
      {
        label: "任务与数据",
        value: `${getTaskFamilyLabel(taskType)} · ${formatAlgorithmDisplay(dataset)}`,
        detail: dataSource,
      },
      {
        label: "攻防配置",
        value: defenseDetail,
        detail: defenseEvents.length ? `${defenseEvents.length} 轮防御证据` : "等待防御事件",
      },
      {
        label: "证据强度",
        value: history.length ? `${history.length} 轮 / ${formatPercent(finalScore)}` : "暂无曲线",
        detail: participation.length ? `${participation.length} 轮客户端状态` : "暂无客户端事件",
      },
    ],
    tags,
  };
}

function resolvePrimaryScore(taskType, latest) {
  if (taskType === "reid") {
    return Number(latest.map || latest.rank1 || 0);
  }
  if (taskType === "ulip3d") {
    return Number(latest.test_accuracy || 0);
  }
  if (Number(latest.test_accuracy || 0) > 0) {
    return Number(latest.test_accuracy || 0);
  }
  return Number(latest.detection_accuracy || 0);
}

function getAlgorithmFamilyLabel(algorithm) {
  const familyMap = {
    fedavg: "基础基线",
    fedprox: "基础基线",
    scaffold: "基础基线",
    feddyn: "正则校正",
    fednova: "基础基线",
    fedadam: "服务端优化",
    fedyogi: "服务端优化",
    fedbn: "个性化/域泛化",
    ditto: "个性化基线",
    pfedme: "个性化基线",
    fedproto: "原型基线",
    feddf: "蒸馏基线",
    fedkd: "蒸馏基线",
    fedcvc: "低参与异构",
    feddtc: "低参与异构",
    rfl_nlcp: "低参与异构",
    fedcads: "蒸馏对齐",
    gfed_hsam: "联邦优化",
    a_fedpd: "参考基线",
    a_fedpdsam: "参考基线",
    fedspeed: "参考基线",
    fedsmoo: "参考基线",
    fedlesam_d: "参考基线",
    fedgloss: "参考基线",
    feddc: "非IID基线",
    fedvra: "非IID基线",
    fedvarp: "非IID基线",
    fedtoga: "非IID基线",
    fedgkd_p: "蒸馏基线",
    fedfld: "蒸馏基线",
    co_evo: "ReID 域泛化",
    moon: "ReID 基线",
    mixstyle: "域泛化基线",
    crossstyle: "域泛化基线",
    fedreid: "ReID 基线",
    fedpav: "ReID 基线",
    snr: "域泛化基线",
    dacs: "域泛化基线",
    sscu: "域泛化基线",
    fedulip: "3D 多模态",
    ulip: "3D 预训练",
    pointclip: "3D 多模态基线",
    fedkgcoop: "3D 多模态基线",
    fedvpt: "3D 多模态基线",
    fedtpg: "3D 多模态基线",
    fedcocoop: "3D 多模态基线",
    fedmaple: "3D 多模态基线",
    fedclip: "3D 多模态基线",
    fedmvp: "3D 多模态基线",
    vert: "鲁棒防御",
    krum: "鲁棒防御",
    multi_krum: "鲁棒防御",
    median: "鲁棒防御",
    trimmed_mean: "鲁棒防御",
    fldetector: "鲁棒防御",
    flbeeline: "可信筛选",
    fltrust: "鲁棒防御",
    flame: "鲁棒防御",
  };
  return familyMap[algorithm] || "扩展算法";
}

function getTaskFamilyLabel(taskType) {
  const taskMap = {
    vision_classification: "视觉分类",
    defense_demo: "攻防演示",
    reid: "行人重识别",
    ulip3d: "3D 多模态",
  };
  return taskMap[taskType] || taskType || "通用任务";
}

function formatAlgorithmDisplay(value) {
  const displayMap = {
    fedavg: "FedAvg",
    fedprox: "FedProx",
    scaffold: "SCAFFOLD",
    feddyn: "FedDyn",
    fednova: "FedNova",
    fedadam: "FedAdam",
    fedyogi: "FedYogi",
    fedbn: "FedBN",
    ditto: "Ditto",
    pfedme: "pFedMe",
    fedproto: "FedProto",
    feddf: "FedDF",
    fedkd: "FedKD",
    fedcvc: "FedCVC",
    feddtc: "FedDTC",
    rfl_nlcp: "RFL-NLCP",
    fedcads: "FedCADS",
    gfed_hsam: "GFed-HSAM",
    a_fedpd: "A-FedPD",
    a_fedpdsam: "A-FedPDSAM",
    fedspeed: "FedSpeed",
    fedsmoo: "FedSMOO",
    fedlesam_d: "FedLESAM-D",
    fedgloss: "FedGLOSS",
    feddc: "FedDC",
    fedvra: "FedVRA",
    fedvarp: "FedVARP",
    fedtoga: "FedTOGA",
    fedgkd_p: "FedGKD-P",
    fedfld: "FedFLD",
    co_evo: "CO-EVO",
    moon: "MOON",
    mixstyle: "MixStyle",
    crossstyle: "CrossStyle",
    fedreid: "FedReID",
    fedpav: "FedPav",
    snr: "SNR",
    dacs: "DACS",
    sscu: "SSCU",
    fedulip: "FedULIP",
    ulip: "ULIP",
    pointclip: "PointCLIP",
    fedkgcoop: "FedKgCoOp",
    fedvpt: "FedVPT",
    fedtpg: "FedTPG",
    fedcocoop: "FedCoCoOp",
    fedmaple: "FedMaPLe",
    fedclip: "FedCLIP",
    fedmvp: "FedMVP",
    vert: "VERT",
    krum: "Krum",
    multi_krum: "Multi-Krum",
    median: "Median",
    trimmed_mean: "Trimmed Mean",
    fldetector: "FLDetector",
    flbeeline: "FLBeeline",
    fltrust: "FLTrust",
    flame: "FLAME",
    feature_poison: "FeaturePoison",
    featurepoison: "FeaturePoison",
    gn: "GN",
    mr: "MR",
    agr: "AGR",
    alie: "ALIE",
    cifar10: "CIFAR-10",
    cifar100: "CIFAR-100",
    mnist: "MNIST",
    ag_news: "AG News",
    market1501: "Market1501",
    modelnet40: "ModelNet40",
    none: "无",
  };
  return displayMap[value] || String(value || "-");
}

function computeDefenseBreakdown(defenseEvent, participationEvents) {
  const round = Number(defenseEvent?.round || 0);
  const participation = (participationEvents || []).find((item) => Number(item?.round || 0) === round) || {};
  const malicious = Array.isArray(participation.malicious) ? participation.malicious : [];
  const filtered = Array.isArray(defenseEvent?.filtered_clients) ? defenseEvent.filtered_clients : [];
  const filteredSet = new Set(filtered.map((item) => Number(item)));
  const missedMalicious = malicious.filter((item) => !filteredSet.has(Number(item)));
  return { missedMalicious };
}

function renderDefenseCards(defenseEvents, participationEvents = []) {
  if (!defenseEvents.length) {
    return renderEmptyState("暂无每轮防御结果，真实模式下会随 round_complete / defense_detection 事件实时更新。");
  }
  const latest = defenseEvents
    .slice()
    .sort((a, b) => Number(b.round || 0) - Number(a.round || 0))[0];
  const breakdown = computeDefenseBreakdown(latest, participationEvents);
  return `
    <div class="list defense-card-grid">
      <div class="list-item defense-metric-card">
        <strong>可信客户端</strong>
        <p class="metric-value">${escapeHtml(formatClientSummary(latest.trusted_clients))}</p>
        <p class="defense-card-footnote">第 ${latest.round} 轮</p>
      </div>
      <div class="list-item defense-metric-card">
        <strong>过滤客户端</strong>
        <p class="metric-value">${escapeHtml(formatClientSummary(latest.filtered_clients))}</p>
        <p class="defense-card-footnote">第 ${latest.round} 轮</p>
      </div>
      <div class="list-item defense-metric-card">
        <strong>漏检恶意</strong>
        <p class="metric-value">${escapeHtml(formatClientSummary(breakdown.missedMalicious))}</p>
        <p class="defense-card-footnote">第 ${latest.round} 轮 · malicious - filtered</p>
      </div>
      <div class="list-item defense-metric-card">
        <strong>检测精度</strong>
        <p class="metric-value">${escapeHtml(formatDetectionAccuracy(latest.detection_accuracy || 0, { defense: state.currentExperiment?.defense, attack: state.currentExperiment?.attack }))}</p>
        <p class="defense-card-footnote">第 ${latest.round} 轮</p>
      </div>
    </div>
  `;
}

function formatClientSummary(clients) {
  const normalized = clients || [];
  if (!normalized.length) {
    return "无";
  }
  return `${normalized.slice(0, 8).join(", ")}${normalized.length > 8 ? "…" : ""} (${normalized.length} 个)`;
}

function renderComparisonLineChart(items) {
  if (!items.length) {
    return renderEmptyState("暂无可对比的趋势数据。");
  }
  const metricConfigMap = {
    accuracy: {
      axisLabel: "Accuracy",
      description: "展示不同算法在训练过程中的精度收敛趋势。",
      seriesBuilder: (item) => {
        const historyValues = item.history?.map((point) => point.test_accuracy).filter((value) => value != null) || [];
        if (historyValues.length) return historyValues;
        if (item.series?.accuracy?.length) return item.series.accuracy;
        return synthesizeSeries(item.final_accuracy || 0.7, item.best_accuracy || item.final_accuracy || 0.7, 8);
      },
    },
    loss: {
      axisLabel: "Loss",
      description: "展示不同算法在训练过程中的损失下降趋势。",
      seriesBuilder: (item) => {
        const historyValues = item.history?.map((point) => point.train_loss).filter((value) => value != null) || [];
        if (historyValues.length) return historyValues;
        if (item.series?.loss?.length) return item.series.loss;
        return synthesizeLossSeries(item.best_loss || 0.3, 8);
      },
    },
    robustness: {
      axisLabel: "Detection Accuracy",
      description: "这里展示的是防御检测精度，也就是系统识别异常/恶意客户端的能力；普通分类或 ReID 任务如果没有防御分支，这条曲线就不会有真实值。",
      seriesBuilder: (item) => {
        const historyValues = item.history?.map((point) => point.detection_accuracy).filter((value) => value != null && Number(value) > 0) || [];
        if (historyValues.length) return historyValues;
        const seriesValues = item.series?.robustness?.filter((value) => value != null && Number(value) > 0) || [];
        if (seriesValues.length) return seriesValues;
        return [];
      },
    },
  };
  const metricConfig = metricConfigMap[state.resultsMetric] || metricConfigMap.accuracy;
  const experiments = items
    .slice(0, 5)
    .map((item) => ({
      label: item.experiment_name,
      color: experimentColor(item.algorithm || item.experiment_id),
      values: metricConfig.seriesBuilder(item),
      rounds: item.history?.map((point) => point.round) || item.series?.rounds || [],
    }))
    .filter((item) => item.values.length);
  if (!experiments.length) {
    return renderEmptyState(
      state.resultsMetric === "robustness"
        ? "当前筛选结果里没有可用的防御检测精度。请选择带攻击/防御的实验，或切换到 Accuracy / Loss。"
        : "当前筛选结果里没有可绘制的历史曲线。"
    );
  }
  const longestRounds = experiments.reduce((best, item) => (item.rounds.length > best.length ? item.rounds : best), []);
  const svg = renderMultiLineSvg({
    width: 1040,
    height: 320,
    xValues: longestRounds.length ? longestRounds : experiments[0].values.map((_, index) => index + 1),
    xAxisLabel: "对比轮次",
    yAxisLabel: metricConfig.axisLabel,
    series: experiments.map((item) => ({
      key: item.label,
      label: item.label,
      color: item.color,
      values: item.values,
    })),
  });
  return `
    <div class="line-chart-header">
      <div class="line-chart-meta">
        <p class="muted">${metricConfig.description}</p>
      </div>
      <div class="line-chart-legend">
        ${experiments
          .map((item) => `<span class="legend-item"><i style="background:${item.color}"></i>${escapeHtml(shorten(item.label, 18))}</span>`)
          .join("")}
      </div>
    </div>
    <div class="line-chart-svg">${svg}</div>
  `;
}

function renderAblationTable(items) {
  if (!items.length) {
    return renderEmptyState("当前筛选条件下没有可展示的实验结果。");
  }
  return `
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>实验</th>
            <th>数据来源</th>
            <th>蒸馏分支</th>
            <th>防御分支</th>
            <th>算法</th>
            <th>最终精度</th>
            <th>检测精度</th>
          </tr>
        </thead>
        <tbody>
          ${items
            .slice(0, 5)
            .map(
              (item, index) => `
                <tr>
                  <td>${escapeHtml(item.experiment_name)}</td>
                  <td>${escapeHtml(renderResultSourceBadge(item))}</td>
                  <td>${index % 2 === 0 ? "开启" : "关闭"}</td>
                  <td>${index % 3 === 0 ? "开启" : "关闭"}</td>
                  <td>${escapeHtml(item.algorithm)}</td>
                  <td>${formatPercent(item.final_accuracy)}</td>
                  <td>${formatDetectionAccuracy(item.detection_accuracy || 0, item)}</td>
                </tr>
              `
            )
            .join("")}
        </tbody>
      </table>
    </div>
  `;
}

function renderResultSourceBadge(item) {
  const source = renderDataSourceLabel(item?.data_source);
  const mode = item?.training_mode || defaultTrainingModeForSource(item?.data_source);
  return `${source} / ${mode}`;
}

function renderResultAgentPanel(workflowSnapshot, experimentId) {
  const result = workflowSnapshot?.result_analysis;
  if (!result) {
    return `
      <section class="panel result-agent-panel result-agent-pending">
        <div class="result-agent-panel-heading"><div><p class="eyebrow">结果分析智能体</p><h3>等待一条完整的证据链</h3></div></div>
        <p class="muted">实验完成后，这里会显示当前运行的观察、结论支持度、限制和下一步补充实验，不会把一次运行包装成算法优越性结论。</p>
      </section>
    `;
  }
  const analysis = result.analysis || {};
  const renderList = (items, empty = "暂无") => {
    const values = Array.isArray(items) && items.length ? items : [empty];
    return `<ul class="result-agent-list">${values.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>`;
  };
  const evidenceCount = (result.evidence_refs || []).length;
  return `
    <section class="panel result-agent-panel">
      <div class="result-agent-panel-heading">
        <div>
          <p class="eyebrow">结果分析智能体</p>
          <h3>把指标变成有边界的实验结论</h3>
          <p class="muted">当前分析只针对本次运行，不执行历史实验比较。</p>
        </div>
        <div class="result-agent-panel-actions">
          <span class="status-badge ok">${escapeHtml(result.claim_level || "descriptive_only")}</span>
          <span class="metric-tag">证据 ${evidenceCount} 条</span>
          ${experimentId && state.backendAvailable ? `<button class="secondary-button" id="regenerate-result-analysis" type="button">重新分析</button>` : ""}
        </div>
      </div>
      <div class="result-agent-columns">
        <article><h4>本次实验观察</h4>${renderList(analysis.observed_outcome)}</article>
        <article><h4>支持的结论</h4>${renderList(analysis.supported_claims)}</article>
        <article><h4>不能推出</h4>${renderList(analysis.unsupported_claims)}</article>
        <article><h4>实验限制</h4>${renderList(analysis.limitations)}</article>
        <article><h4>异常说明</h4>${renderList((analysis.anomalies || []).map((item) => item.message || item.code || item))}</article>
        <article><h4>建议补充实验</h4>${renderList(analysis.next_experiments)}</article>
      </div>
    </section>
  `;
}

function renderResultsSourceHint(items, backendAvailable) {
  if (!items.length) {
    return `<p class="muted">当前筛选结果为空。后端在线时本页默认只展示真实结果；后端不可用时会回退到模拟数据。</p>`;
  }
  const backendCount = items.filter((item) => isBackendExperimentSource(item.data_source)).length;
  const mockCount = items.length - backendCount;
  if (!backendAvailable) {
    return `<p class="muted">当前后端不可用，本页已回退到模拟数据视图。</p>`;
  }
  if (backendCount > 0 && mockCount === 0) {
    return `<p class="muted">当前筛选结果共 ${items.length} 条，全部来自后端实验结果。</p>`;
  }
  if (backendCount > 0 && mockCount > 0) {
    return `<p class="muted">当前筛选结果共 ${items.length} 条，其中后端实验结果 ${backendCount} 条，前端模拟结果 ${mockCount} 条。</p>`;
  }
  return `<p class="muted">当前筛选结果共 ${items.length} 条，但都不是后端真实实验结果。</p>`;
}

function isBackendExperimentSource(source) {
  return source === "real" || source === "backend_synthetic";
}

function renderDataSourceLabel(source) {
  if (source === "real") return "真实文件数据";
  if (source === "backend_synthetic") return "后端模拟数据";
  return "前端模拟数据";
}

function defaultTrainingModeForSource(source) {
  if (source === "real") return "real_federated";
  if (source === "backend_synthetic") return "pytorch_federated";
  return "synthetic";
}

function renderMultiLineSvg({ width, height, xValues, series, xAxisLabel = "横轴", yAxisLabel = "纵轴", yPaddingRatio = 0.08 }) {
  const margin = { top: 18, right: 20, bottom: 48, left: 64 };
  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;
  const normalizedSeries = series.map((item) => ({
    ...item,
    values: item.values.map((value) => {
      const num = Number(value);
      return Number.isFinite(num) ? num : null;
    }),
  }));
  const allValues = normalizedSeries.flatMap((item) => item.values).filter((value) => value != null);
  let minValue = Math.min(0, ...allValues);
  let maxValue = Math.max(0.01, ...allValues);
  let span = maxValue - minValue || 1;
  if (allValues.length) {
    const padding = span * Math.max(0, yPaddingRatio);
    minValue -= padding;
    maxValue += padding;
    span = maxValue - minValue || 1;
  }
  const xStep = xValues.length > 1 ? innerWidth / (xValues.length - 1) : innerWidth;
  const yTicks = 4;

  const yScale = (value) => margin.top + innerHeight - ((value - minValue) / span) * innerHeight;
  const xScale = (index) => margin.left + index * xStep;

  const gridLines = Array.from({ length: yTicks + 1 }, (_, index) => {
    const y = margin.top + (innerHeight / yTicks) * index;
    return `<line x1="${margin.left}" y1="${y}" x2="${width - margin.right}" y2="${y}" class="chart-grid-line" />`;
  }).join("");

  const yLabels = Array.from({ length: yTicks + 1 }, (_, index) => {
    const value = maxValue - (span / yTicks) * index;
    const y = margin.top + (innerHeight / yTicks) * index + 5;
    return `<text x="${margin.left - 8}" y="${y}" text-anchor="end" class="chart-axis-label">${value.toFixed(2)}</text>`;
  }).join("");

  const maxXticks = 10;
  const xTickInterval = Math.max(1, Math.ceil(xValues.length / maxXticks));
  const xLabels = xValues
    .filter((_, index) => index % xTickInterval === 0 || index === xValues.length - 1)
    .map((value, index) => {
      const realIndex = xValues.indexOf(value);
      const x = xScale(realIndex);
      return `<text x="${x}" y="${height - margin.bottom + 18}" text-anchor="middle" class="chart-axis-label">${value}</text>`;
    })
    .join("");

  const lines = normalizedSeries
    .map((item) => {
      const points = item.values
        .map((value, index) => (value == null ? null : `${xScale(index)},${yScale(value)}`))
        .filter(Boolean)
        .join(" ");
      const circles = item.values
        .map((value, index) => (value == null ? "" : `<circle cx="${xScale(index)}" cy="${yScale(value)}" r="3" fill="${item.color}" />`))
        .join("");
      return points ? `<polyline fill="none" stroke="${item.color}" stroke-width="2.5" points="${points}" />${circles}` : circles;
    })
    .join("");

  return `
    <svg viewBox="0 0 ${width} ${height}" class="line-chart" role="img" aria-label="多折线图">
      ${gridLines}
      <line x1="${margin.left}" y1="${margin.top}" x2="${margin.left}" y2="${height - margin.bottom}" class="chart-axis-line" />
      <line x1="${margin.left}" y1="${height - margin.bottom}" x2="${width - margin.right}" y2="${height - margin.bottom}" class="chart-axis-line" />
      ${yLabels}
      ${xLabels}
      ${lines}
      <text x="${width / 2}" y="${height - 4}" text-anchor="middle" class="chart-axis-caption">${escapeHtml(xAxisLabel)}</text>
      <text x="20" y="${height / 2}" text-anchor="middle" class="chart-axis-caption" transform="rotate(-90 20 ${height / 2})">${escapeHtml(yAxisLabel)}</text>
    </svg>
  `;
}

function renderBarChartSvg(items, valueKey, labelKey, xAxisLabel = "横轴", yAxisLabel = "纵轴") {
  const width = 860;
  const height = 304;
  const margin = { top: 18, right: 18, bottom: 58, left: 58 };
  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;
  const values = items.map((item) => Number(item[valueKey] || 0));
  const maxValue = Math.max(...values, 1);
  const step = innerWidth / items.length;
  const barWidth = step * 0.58;
  const yTicks = 4;

  const gridLines = Array.from({ length: yTicks + 1 }, (_, index) => {
    const y = margin.top + (innerHeight / yTicks) * index;
    return `<line x1="${margin.left}" y1="${y}" x2="${width - margin.right}" y2="${y}" class="chart-grid-line" />`;
  }).join("");

  const yLabels = Array.from({ length: yTicks + 1 }, (_, index) => {
    const value = maxValue - (maxValue / yTicks) * index;
    const y = margin.top + (innerHeight / yTicks) * index + 4;
    return `<text x="${margin.left - 8}" y="${y}" text-anchor="end" class="chart-axis-label">${value.toFixed(2)}</text>`;
  }).join("");

  const bars = items.map((item, index) => {
    const value = Number(item[valueKey] || 0);
    const heightValue = (value / maxValue) * innerHeight;
    const x = margin.left + step * index + (step - barWidth) / 2;
    const y = margin.top + innerHeight - heightValue;
    return `
      <rect x="${x}" y="${y}" width="${barWidth}" height="${heightValue}" rx="8" class="bar-chart-bar" />
      <text x="${x + barWidth / 2}" y="${height - margin.bottom + 16}" text-anchor="middle" class="chart-axis-label">${escapeHtml(shorten(item[labelKey], 12))}</text>
    `;
  }).join("");

  return `
    <div class="line-chart-svg">
      <svg viewBox="0 0 ${width} ${height}" class="line-chart" role="img" aria-label="柱状图">
        ${gridLines}
        <line x1="${margin.left}" y1="${margin.top}" x2="${margin.left}" y2="${height - margin.bottom}" class="chart-axis-line" />
        <line x1="${margin.left}" y1="${height - margin.bottom}" x2="${width - margin.right}" y2="${height - margin.bottom}" class="chart-axis-line" />
        ${yLabels}
        ${bars}
        <text x="${width / 2}" y="${height - 4}" text-anchor="middle" class="chart-axis-caption">${escapeHtml(xAxisLabel)}</text>
        <text x="18" y="${height / 2}" text-anchor="middle" class="chart-axis-caption" transform="rotate(-90 18 ${height / 2})">${escapeHtml(yAxisLabel)}</text>
      </svg>
    </div>
  `;
}

function synthesizeSeries(finalValue, bestValue, points = 8) {
  return Array.from({ length: points }, (_, index) => {
    const progress = (index + 1) / points;
    const eased = finalValue * 0.72 + (bestValue - finalValue * 0.72) * progress;
    const wobble = Math.sin(index * 0.8) * 0.015;
    return Number((eased + wobble).toFixed(3));
  });
}

function experimentColor(seedText) {
  const palette = ["#0f6b62", "#d86f45", "#4f46e5", "#7c3aed", "#b45309", "#2563eb"];
  let hash = 0;
  for (const ch of String(seedText)) {
    hash = (hash * 31 + ch.charCodeAt(0)) >>> 0;
  }
  return palette[hash % palette.length];
}

function calculateRemainingTime(currentRound, totalRounds, createdAt) {
  const remainingRounds = Math.max((totalRounds || 0) - (currentRound || 0), 0);
  if (remainingRounds === 0) {
    return "0 秒";
  }
  const createdAtMs = parseBackendDate(createdAt);
  const completedRounds = Math.max(currentRound || 0, 0);
  let seconds = remainingRounds * 0.3;
  if (Number.isFinite(createdAtMs) && completedRounds > 0) {
    const elapsedSeconds = Math.max((Date.now() - createdAtMs) / 1000, 0);
    const avgRoundSeconds = elapsedSeconds / completedRounds;
    if (Number.isFinite(avgRoundSeconds) && avgRoundSeconds > 0) {
      seconds = remainingRounds * avgRoundSeconds;
    }
  }
  if (seconds < 60) {
    return `${seconds.toFixed(0)} 秒`;
  }
  const minutes = Math.floor(seconds / 60);
  const restSeconds = Math.round(seconds % 60);
  return `${minutes} 分 ${restSeconds} 秒`;
}

function parseBackendDate(value) {
  if (!value) {
    return Number.NaN;
  }
  const raw = String(value);
  const normalized = /z$|[+-]\d{2}:\d{2}$/i.test(raw) ? raw : `${raw}Z`;
  const timestamp = new Date(normalized).getTime();
  return Number.isFinite(timestamp) ? timestamp : Number.NaN;
}

function renderMonitorStatusText(status, currentRound, totalRounds, createdAt) {
  if (status === "completed") {
    return "已完成";
  }
  if (status === "stopped") {
    return "已停止";
  }
  return `预计剩余 ${calculateRemainingTime(currentRound, totalRounds, createdAt)}`;
}

function buildMonitorLogLines(logs) {
  if (!Array.isArray(logs) || !logs.length) {
    return [];
  }
  return logs.map((entry) => {
    const payload = entry.payload || {};
    const timeText = entry.created_at ? `[${new Date(entry.created_at).toLocaleTimeString("zh-CN")}] ` : "";
    if (entry.event_type === "round_complete") {
      return `${timeText}[round_complete] round=${payload.round} train_loss=${payload.train_loss} test_accuracy=${payload.test_accuracy}`;
    }
    if (entry.event_type === "client_participation") {
      return `${timeText}[client_participation] round=${payload.round} participating=${(payload.participating || []).length} malicious=${(payload.malicious || []).length}`;
    }
    if (entry.event_type === "defense_detection") {
      return `${timeText}[defense_detection] round=${payload.round} detection_accuracy=${payload.detection_accuracy}`;
    }
    if (entry.event_type === "training_mode_selected") {
      return `${timeText}[training_mode_selected] ${payload.message || "training mode selected"}`;
    }
    if (entry.event_type === "security_observation") {
      return `${timeText}安全监控：第 ${payload.round || "-"} 轮，状态：${workflowStatusText(payload.status)}，建议动作：${workflowActionText(payload.recommended_action)}。`;
    }
    if (entry.event_type === "runtime_agent_check") {
      return `${timeText}运行中周期检查：第 ${payload.round || "-"} 轮，状态：${workflowStatusText(payload.status)}，处置：${workflowActionText(payload.control_action)}。`;
    }
    if (entry.event_type === "agent_update") {
      return `${timeText}智能体更新：${payload.agent_name || "智能体"}，阶段：${workflowStageText(payload.stage)}，状态：${workflowStatusText(payload.status)}。`;
    }
    if (entry.event_type === "result_analysis_completed" || entry.event_type === "result_analysis_regenerated") {
      return `${timeText}[${entry.event_type}] ${payload.message || "result analysis ready"}`.trim();
    }
    if (entry.event_type === "completed" || entry.event_type === "stopped" || entry.event_type === "failed") {
      return `${timeText}[${entry.event_type}] ${payload.message || ""}`.trim();
    }
    return `${timeText}[${entry.event_type}]`;
  });
}

function getAvailableMonitorMetrics(experiment) {
  const taskType = experiment?.task_type || "";
  if (taskType === "reid") {
    return [
      { key: "map", label: "mAP" },
      { key: "rank1", label: "Rank-1" },
      { key: "train_loss", label: "Train Loss" },
    ];
  }
  const metrics = [
    { key: "train_loss", label: "Train Loss" },
    { key: "test_accuracy", label: "Test Accuracy" },
  ];
  if (taskType === "defense_demo" || experiment?.defense !== "none" || experiment?.attack !== "none") {
    metrics.push({ key: "detection_accuracy", label: "Detection Accuracy" });
  }
  return metrics;
}

function renderMonitorMetricHint(experiment) {
  if (experiment?.task_type === "reid") {
    return "";
  }
  return `<p class="muted monitor-metric-hint">当前任务为 ${escapeHtml(experiment?.task_type || "vision_classification")}，因此本页不展示仅适用于 ReID 检索任务的 mAP / Rank-1。</p>`;
}

function renderMonitorDataSource(experiment, metrics) {
  const config = experiment?.config_json || {};
  const inferredDataSource = metrics?.history?.length ? "backend_synthetic" : "mock";
  const dataSource = config.data_source || inferredDataSource;
  const trainingMode = config.training_mode || defaultTrainingModeForSource(dataSource);
  return `${renderDataSourceLabel(dataSource)} / ${trainingMode}`;
}

function renderSelectField(name, label, options, currentValue) {
  return `
    <div class="form-field">
      <label for="${escapeHtml(name)}">${escapeHtml(label)}</label>
      <select id="${escapeHtml(name)}" name="${escapeHtml(name)}" class="select">
        ${options
          .map(([value, text]) => `<option value="${escapeHtml(value)}" ${value === currentValue ? "selected" : ""}>${escapeHtml(text)}</option>`)
          .join("")}
      </select>
    </div>
  `;
}

function renderNumberField(name, label, currentValue, step) {
  return `
    <div class="form-field">
      <label for="${escapeHtml(name)}">${escapeHtml(label)}</label>
      <input id="${escapeHtml(name)}" name="${escapeHtml(name)}" class="input" type="number" value="${currentValue}" step="${step}" />
    </div>
  `;
}

function renderEmptyState(message) {
  const template = document.querySelector("#empty-state-template");
  const node = template.content.cloneNode(true);
  node.querySelector(".muted").textContent = message;
  const wrapper = document.createElement("div");
  wrapper.appendChild(node);
  return wrapper.innerHTML;
}

function clearApiCache() {
  apiResponseCache.clear();
}

function clonePayload(value) {
  if (value === undefined || value === null) {
    return value;
  }
  if (typeof structuredClone === "function") {
    return structuredClone(value);
  }
  return JSON.parse(JSON.stringify(value));
}

const api = {
  getOverview: () => fetchWithFallback("/api/overview", () => mockDb.overview, { cacheMs: API_CACHE_TTL_MS }),
  getOverviewCapabilityMap: () => fetchWithFallback("/api/overview/capability-map", () => mockDb.capabilityMap, { cacheMs: API_CACHE_TTL_MS }),
  getModules: () => fetchWithFallback("/api/modules", () => mockDb.modules, { cacheMs: API_CACHE_TTL_MS }),
  getModuleById: (id) => fetchWithFallback(`/api/modules/${id}`, () => mockDb.modules.find((item) => item.id === id), { cacheMs: API_CACHE_TTL_MS }),
  getModuleReferences: (id) => fetchWithFallback(`/api/modules/${id}/references`, () => mockDb.moduleReferences[id], { cacheMs: API_CACHE_TTL_MS }),
  getAlgorithms: () => fetchWithFallback("/api/algorithms", () => mockDb.algorithms, { cacheMs: API_CACHE_TTL_MS }),
  getDatasets: () => fetchWithFallback("/api/datasets", () => mockDb.datasets, { cacheMs: API_CACHE_TTL_MS }),
  getOptimizers: () => fetchWithFallback("/api/optimizers", () => mockDb.optimizers, { cacheMs: API_CACHE_TTL_MS }),
  getDefenses: () => fetchWithFallback("/api/defenses", () => mockDb.defenses, { cacheMs: API_CACHE_TTL_MS }),
  getAttacks: () => fetchWithFallback("/api/attacks", () => mockDb.attacks, { cacheMs: API_CACHE_TTL_MS }),
  getExperimentConfigOptions: (moduleId) => fetchWithFallback(`/api/experiment-config-options?module_id=${encodeURIComponent(moduleId || "")}`, () => getMockConfigOptions(moduleId), { cacheMs: API_CACHE_TTL_MS }),
  getExperimentTemplates: () => fetchWithFallback("/api/experiments/templates", () => mockDb.templates, { cacheMs: API_CACHE_TTL_MS }),
  parseExperimentConfig: async (payload) => {
    const response = await fetch(`${API_BASE}/api/experiment-config-assistant/parse`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!response.ok) {
      let detail = `HTTP ${response.status}`;
      try {
        const errorPayload = await response.json();
        detail = errorPayload.detail || detail;
      } catch (error) {
        // Keep the HTTP status when the error response is not JSON.
      }
      throw new Error(detail);
    }
    state.backendAvailable = true;
    syncBackendBadge();
    return response.json();
  },
  createExperiment: (config) =>
    fetchWithFallback(
      "/api/experiments",
      () => {
        const id = `exp_demo_${String(mockDb.results.length + 1).padStart(3, "0")}`;
        const experiment = {
          id,
          name: `${config.algorithm} 在 ${config.dataset} 上的演示实验`,
          ...config,
          status: "draft",
          current_round: 0,
          best_accuracy: 0.0,
        };
        mockDb.experiments.unshift(experiment);
        return experiment;
      },
      {
        method: "POST",
        body: config,
      }
    ),
  startExperiment: (id) =>
    fetchWithFallback(`/api/experiments/${id}/start`, () => {
      const experiment = mockDb.experiments.find((item) => item.id === id);
      if (experiment) {
        experiment.status = "running";
      }
      return { status: "started", workflow_status: "running" };
    }, {
      method: "POST",
    }),
  runPreflight: (id) =>
    fetchWithFallback(
      `/api/experiments/${id}/preflight`,
      () => {
        const experiment = mockDb.experiments.find((item) => item.id === id);
        if (experiment) {
          experiment.status = "ready_to_start";
        }
        return getMockWorkflow(id);
      },
      { method: "POST" }
    ),
  getWorkflow: (id) =>
    fetchWithFallback(`/api/experiments/${id}/workflow`, () => getMockWorkflow(id)),
  regenerateResultAnalysis: (id) =>
    fetchWithFallback(
      `/api/experiments/${id}/result-analysis/regenerate`,
      () => getMockWorkflow(id).result_analysis,
      { method: "POST" }
    ),
  getExperiments: (query = {}) =>
    fetchWithFallback(
      `/api/experiments${toQueryString(query)}`,
      () => mockDb.experiments.filter((item) => !query.status || item.status === query.status)
    ),
  getExperiment: (id) =>
    fetchWithFallback(`/api/experiments/${id}`, () => mockDb.experiments.find((item) => item.id === id) || mockDb.experiments[0]),
  getExperimentMetrics: (id) =>
    fetchWithFallback(`/api/experiments/${id}/metrics`, () => mockDb.metrics[id] || mockDb.metrics.exp_demo_001),
  getExperimentLogs: (id, limit = 200) =>
    fetchWithFallback(`/api/experiments/${id}/logs?limit=${limit}`, () => []),
  stopExperiment: (id) =>
    fetchWithFallback(
      `/api/experiments/${id}/stop`,
      () => {
        const experiment = mockDb.experiments.find((item) => item.id === id);
        if (experiment) {
          experiment.status = "stopped";
          experiment.current_round = Number(state.currentExperiment?.current_round || experiment.current_round || 0);
        }
        return { status: "stopped", experiment_id: id };
      },
      {
        method: "POST",
      }
    ),
  getRecentResults: (limit = 10) =>
    fetchWithFallback(`/api/results/recent?limit=${limit}`, () => mockDb.results.slice(0, limit), { cacheMs: 5000 }),
  getResults: (query = {}) =>
    fetchWithFallback(
      `/api/results${toQueryString(query)}`,
      () =>
        mockDb.results.filter((item) => {
          if (query.algorithm && item.algorithm !== query.algorithm) return false;
          if (query.dataset && item.dataset !== query.dataset) return false;
          if (query.task_type && item.task_type !== query.task_type) return false;
          if (query.defense && (item.defense || "none") !== query.defense) return false;
          if (query.attack && (item.attack || "none") !== query.attack) return false;
          if (query.participation_rate && Math.abs((item.participation_rate || 0.1) - Number(query.participation_rate)) > 0.001) return false;
          if (query.seed && (item.seed || 42) !== Number(query.seed)) return false;
          return true;
        })
    ),
  compareResults: (experimentIds) =>
    fetchWithFallback(
      "/api/results/compare",
      () => mockDb.compareResults.filter((item) => experimentIds.includes(item.experiment_id)),
      {
        method: "POST",
        body: { experiment_ids: experimentIds },
      }
    ),
  exportResults: (format, experimentIds = []) =>
    fetchWithFallback(
      `/api/results/export?format=${format}${experimentIds.length ? `&experiment_ids=${experimentIds.join(",")}` : ""}`,
      () => ({
        message: `已生成 ${format.toUpperCase()} 导出任务${experimentIds.length ? `，包含 ${experimentIds.length} 个实验` : "，包含当前筛选结果"}`,
      })
    ),
  getNetworks: (taskType) => fetchWithFallback(`/api/networks?task_type=${taskType || ""}`, () => getMockNetworksForTask(taskType), { cacheMs: API_CACHE_TTL_MS }),
};

async function fetchWithFallback(path, fallbackFactory, options = {}) {
  const requestOptions = {
    method: options.method || "GET",
    headers: {
      "Content-Type": "application/json",
    },
  };

  if (options.body) {
    requestOptions.body = JSON.stringify(options.body);
  }

  const cacheKey = requestOptions.method === "GET" && options.cacheMs ? path : null;
  if (cacheKey) {
    const cached = apiResponseCache.get(cacheKey);
    if (cached && Date.now() - cached.createdAt < options.cacheMs) {
      return clonePayload(cached.data);
    }
  }

  try {
    const response = await fetch(`${API_BASE}${path}`, requestOptions);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    state.backendAvailable = true;
    syncBackendBadge();
    const payload = await response.json();
    if (cacheKey) {
      apiResponseCache.set(cacheKey, { createdAt: Date.now(), data: clonePayload(payload) });
    }
    if (requestOptions.method !== "GET") {
      clearApiCache();
    }
    return payload;
  } catch (error) {
    state.backendAvailable = false;
    syncBackendBadge();
    showAlert("error", "真实数据加载失败，已自动回退到模拟数据。");
  }

  const fallbackPayload = clonePayload(fallbackFactory());
  if (cacheKey) {
    apiResponseCache.set(cacheKey, { createdAt: Date.now(), data: clonePayload(fallbackPayload) });
  }
  if (requestOptions.method !== "GET") {
    clearApiCache();
  }
  return fallbackPayload;
}

function getMockWorkflow(experimentId) {
  const experiment = mockDb.experiments.find((item) => item.id === experimentId) || mockDb.experiments[0] || {};
  const status = experiment.status || "draft";
  const preflightReady = ["ready_to_start", "running", "completed", "analyzed"].includes(status);
  const finished = ["completed", "analyzed"].includes(status);
  const running = status === "running";
  const runtimeInterval = Number(experiment.runtime_agent_check_interval_rounds || 3);
  const runtimeCheckCount = Math.floor(Number(experiment.current_round || 0) / Math.max(runtimeInterval, 1));
  return {
    experiment_id: experiment.id || experimentId,
    experiment_status: status,
    agents: [
      {
        id: "mock-scenario-agent",
        agent_name: "实验场景智能体",
        stage: "scenario_configuration",
        status: "completed",
        input_hash: "mock",
        output: { summary: { message: "配置草案已归一化。" } },
        tool_calls: [{ tool: "catalog.read", status: "ok" }, { tool: "config.resolve", status: "ok" }],
      },
      {
        id: "mock-operations-agent",
        agent_name: "运行保障智能体",
        stage: preflightReady ? (finished ? "finalize" : (runtimeCheckCount ? "runtime" : "preflight")) : "preflight",
        status: preflightReady ? "completed" : "pending",
        input_hash: "mock",
        output: { round: runtimeCheckCount * runtimeInterval, summary: { message: preflightReady ? (runtimeCheckCount ? `已完成 ${runtimeCheckCount} 次运行中保障检查。` : "运行条件已检查。") : "等待运行预检。" } },
        tool_calls: [],
      },
      {
        id: "mock-security-agent",
        agent_name: "安全监控智能体",
        stage: running ? "runtime" : (preflightReady ? (runtimeCheckCount ? "runtime" : "preflight_baseline") : "preflight_baseline"),
        status: running ? "running" : (preflightReady ? "completed" : "pending"),
        input_hash: "mock",
        output: { round: runtimeCheckCount * runtimeInterval, status: "normal", recommended_action: "continue" },
        tool_calls: [],
      },
      {
        id: "mock-result-agent",
        agent_name: "结果分析智能体",
        stage: "result_analysis",
        status: finished ? "completed" : "pending",
        input_hash: "mock",
        output: { claim_level: "descriptive_only" },
        tool_calls: [],
      },
    ],
    operations_report: preflightReady
      ? { stage: finished ? "finalize" : (runtimeCheckCount ? "runtime" : "preflight"), status: "passed", findings: [], manifest: { seed: experiment.seed || 42 } }
      : null,
    security_report: preflightReady
      ? { status: "normal", agent_security_findings: [], fl_security_findings: [], evidence_refs: [], recommended_action: "continue", observation_count: runtimeCheckCount }
      : null,
    runtime_agent_check_interval_rounds: runtimeInterval,
    runtime_check_count: runtimeCheckCount,
    result_analysis: finished
      ? {
          experiment_id: experiment.id || experimentId,
          status: "completed",
          claim_level: "descriptive_only",
          analysis: {
            observed_outcome: ["本次运行的指标和曲线已生成。"],
            supported_claims: ["只能描述当前数据、配置和 seed 下的运行表现。"],
            unsupported_claims: ["不能据此证明算法优于 FedAvg。"],
            limitations: ["当前演示不执行历史实验比较。"],
            anomalies: [],
            next_experiments: ["补充同配置 FedAvg 对照和多 seed 重复。"],
          },
          evidence_refs: [],
        }
      : null,
  };
}

function getMockNetworksForTask(taskType) {
  const byTask = {
    vision_classification: [
      { id: "small_cnn", name: "Small CNN" },
      { id: "resnet18", name: "ResNet-18" },
      { id: "mobilenetv2", name: "MobileNetV2" },
      { id: "lenet5", name: "LeNet-5" },
    ],
    text_classification: [{ id: "textcnn", name: "TextCNN" }],
    defense_demo: [
      { id: "small_cnn", name: "Small CNN" },
      { id: "resnet18", name: "ResNet-18" },
    ],
    reid: [
      { id: "tiny_reid", name: "Tiny ReID" },
      { id: "resnet50", name: "ResNet-50" },
      { id: "vgg16", name: "VGG-16" },
    ],
    ulip3d: [{ id: "pointbert", name: "PointBERT" }],
  };
  return byTask[taskType] || byTask.vision_classification;
}

function getMockConfigOptions(moduleId = "vision_benchmark") {
  const constraints = {
    participation_adaptation: {
      task_types: ["vision_classification", "text_classification"],
      datasets: ["mnist", "cifar10", "cifar100", "tiny_imagenet", "ag_news"],
      splits: ["dirichlet", "pathological"],
      protocols: ["limited_participation"],
      networks: ["small_cnn", "resnet18", "mobilenetv2", "textcnn"],
      algorithms: ["fedcvc", "feddtc", "rfl_nlcp", "fedavg", "fedprox", "scaffold", "feddyn", "a_fedpd", "feddc", "fedvra", "fedvarp", "fedspeed", "fedsmoo", "fedtoga", "gfed_hsam"],
      optimizers: ["sgd"],
      defenses: ["none"],
      attacks: ["none"],
      paper_basis: ["FedCVC", "RFL-NLCP", "FedDTC"],
    },
    optimization_control: {
      task_types: ["vision_classification"],
      datasets: ["cifar10", "cifar100", "tiny_imagenet"],
      splits: ["dirichlet"],
      protocols: ["sam_comparison"],
      networks: ["small_cnn", "resnet18", "mobilenetv2"],
      algorithms: ["gfed_hsam", "sam", "esam", "hsam", "fedavg", "feddyn", "fedprox", "scaffold", "a_fedpd", "fedspeed", "fedsmoo", "fedlesam_d", "a_fedpdsam", "fedgloss", "fedvra", "fedvarp", "fedtoga"],
      optimizers: ["sgd", "sam", "esam", "hsam"],
      defenses: ["none"],
      attacks: ["none"],
      paper_basis: ["GFed-HSAM", "SAM", "ESAM", "FedCVC"],
    },
    distillation_alignment: {
      task_types: ["vision_classification"],
      datasets: ["cifar10", "cifar100"],
      splits: ["dirichlet"],
      protocols: ["low_participation_distillation"],
      networks: ["small_cnn", "resnet18", "mobilenetv2"],
      algorithms: ["fedcads", "fedavg", "scaffold", "feddyn", "feddc", "fedvra", "fedgkd_p", "a_fedpd", "fedfld", "feddf", "fedkd"],
      optimizers: ["sgd"],
      defenses: ["none"],
      attacks: ["none"],
      paper_basis: ["FedCADS"],
    },
    robust_aggregation: {
      task_types: ["defense_demo"],
      datasets: ["mnist", "cifar10", "femnist"],
      splits: ["dirichlet"],
      protocols: ["model_poisoning"],
      networks: ["small_cnn", "resnet18"],
      algorithms: ["fedavg", "vert", "krum", "median", "trimmed_mean", "fldetector", "multi_krum", "fltrust", "flame"],
      optimizers: ["sgd"],
      defenses: ["vert", "krum", "multi_krum", "median", "trimmed_mean", "fldetector", "fltrust", "flame"],
      attacks: ["none", "gn", "mr", "agr", "alie", "featurepoison", "fedproto_prototype_attack"],
      paper_basis: ["FLDetector", "Krum", "Median/Trimmed Mean", "VERT"],
    },
    trust_screening: {
      task_types: ["defense_demo"],
      datasets: ["mnist", "fashion_mnist", "emnist", "cifar10", "ag_news"],
      splits: ["dirichlet"],
      protocols: ["topk_trust_screening"],
      networks: ["small_cnn", "resnet18", "textcnn"],
      algorithms: ["flbeeline", "fedavg", "krum", "multi_krum", "median", "flame", "fltrust"],
      optimizers: ["sgd"],
      defenses: ["flbeeline", "krum", "multi_krum", "median", "flame", "fltrust"],
      attacks: ["none", "gn", "mr", "agr", "alie"],
      paper_basis: ["FLBeeline"],
    },
    representation_attack: {
      task_types: ["defense_demo"],
      datasets: ["cifar10"],
      splits: ["iid", "dirichlet"],
      protocols: ["feature_poisoning"],
      networks: ["small_cnn", "resnet18"],
      algorithms: ["featurepoison", "fedproto_prototype_attack", "fedavg", "fedproto"],
      optimizers: ["sgd"],
      defenses: ["none", "vert", "fltrust", "flame"],
      attacks: ["featurepoison", "fedproto_prototype_attack"],
      paper_basis: ["FeaturePoisonAttack"],
    },
    vision_benchmark: {
      task_types: ["vision_classification"],
      datasets: ["mnist", "cifar10", "cifar100", "tiny_imagenet"],
      splits: ["iid", "dirichlet", "pathological"],
      protocols: ["visual_classification"],
      networks: ["small_cnn", "resnet18", "mobilenetv2", "lenet5"],
      algorithms: ["fedavg", "feddyn", "fedcads", "rfl_nlcp", "fedcvc", "feddtc", "gfed_hsam", "fedprox", "scaffold", "fednova", "fedadam", "fedyogi", "fedbn", "ditto", "pfedme", "fedproto", "feddf", "fedkd", "feddc", "fedvra", "fedvarp", "fedgkd_p", "a_fedpd", "fedfld", "fedspeed", "fedsmoo", "fedtoga"],
      optimizers: ["sgd", "sam", "esam", "hsam"],
      defenses: ["none"],
      attacks: ["none"],
      paper_basis: ["FedCVC", "FedCADS", "RFL-NLCP", "GFed-HSAM"],
    },
    reid_generalization: {
      task_types: ["reid"],
      datasets: ["cuhk02", "cuhk03", "msmt17", "market1501"],
      splits: ["domain_as_client"],
      protocols: ["leave_one_domain_out", "multi_source_mixed_test", "source_domain_evaluation"],
      networks: ["tiny_reid", "resnet50", "vgg16"],
      algorithms: ["co_evo", "fedbn", "ditto", "pfedme", "fedproto", "scaffold", "moon", "fedprox", "mixstyle", "crossstyle", "fedreid", "fedpav", "snr", "dacs", "sscu"],
      optimizers: ["sgd"],
      defenses: ["none"],
      attacks: ["none"],
      paper_basis: ["CO-EVO"],
    },
    multimodal_3d: {
      task_types: ["ulip3d"],
      datasets: ["modelnet40", "scanobjectnn", "shapenetcore", "mvtec3d", "mnist3d", "3dimage"],
      splits: ["iid"],
      protocols: ["base2new", "cross_dataset", "domain_abcd"],
      networks: ["pointbert"],
      algorithms: ["fedulip", "fedavg", "ulip", "pointclip", "fedkgcoop", "fedvpt", "fedtpg", "fedcocoop", "fedmaple", "fedclip", "fedmvp"],
      optimizers: ["sgd"],
      defenses: ["none"],
      attacks: ["none"],
      paper_basis: ["FedULIP", "ULIP"],
    },
  };
  const selectedModuleId = constraints[moduleId] ? moduleId : "vision_benchmark";
  const selected = constraints[selectedModuleId];
  const defaults = {
    module_id: selectedModuleId,
    task_type: selected.task_types[0],
    dataset: selected.datasets[0],
    split: selected.splits[0],
    protocol: selected.protocols[0],
    network: selected.networks[0],
    algorithm: selected.algorithms[0],
    optimizer: selected.optimizers.includes("sgd") ? "sgd" : selected.optimizers[0],
    defense: selected.defenses.includes("none") ? "none" : selected.defenses[0],
    attack: selected.attacks.includes("none") ? "none" : selected.attacks[0],
  };
  return { module_id: selectedModuleId, ...selected, defaults, partition_params: {} };
}

function clearMonitorResources() {
  if (monitorTimer) {
    window.clearInterval(monitorTimer);
    monitorTimer = null;
  }
  if (eventSourceRef) {
    eventSourceRef.close();
    eventSourceRef = null;
  }
}

function createMockDatabase() {
  const modules = [
    buildModule("core_engine", "训练编排核心", "联邦训练核心循环与状态管理", "实现 Server/Client/Trainer/Registry 的核心编排", ["总览", "模块能力", "运行监控"], ["core/server.py", "core/client.py", "core/trainer.py"], ["FedAvg", "FedProx", "SCAFFOLD", "FedDyn", "FedNova", "FedAdam", "FedYogi", "FedBN", "Ditto", "pFedMe", "FedProto", "FedDF", "FedKD"]),
    buildModule("participation_adaptation", "低参与异构适配", "低参与率与异构感知补偿", "聚焦 Non-IID、silence table、EMA 异质性与梯度漂移修正", ["模块能力", "实验配置", "运行监控"], ["plugins/algorithms/fedcvc.py", "plugins/algorithms/feddtc.py"], ["FedCVC", "FedDTC", "RFL-NLCP"], ["FedAvg", "FedProx", "SCAFFOLD", "FedDyn", "A-FedPD", "FedDC", "FedVRA", "FedVARP", "FedSpeed", "FedSMOO", "FedTOGA", "GFed-HSAM"]),
    buildModule("optimization_control", "联邦优化控制", "优化器与 sharpness-aware 约束", "支持 SAM/ESAM/HSAM 与 GFed-HSAM", ["模块能力", "实验配置"], ["plugins/optimizers/sam.py", "plugins/optimizers/hsam.py"], ["GFed-HSAM", "SAM", "ESAM", "HSAM"], ["FedAvg", "FedDyn", "FedProx", "SCAFFOLD", "A-FedPD", "FedSpeed", "FedSMOO", "FedLESAM-D", "A-FedPDSAM", "FedGLOSS", "FedVRA", "FedVARP", "FedTOGA"]),
    buildModule("distillation_alignment", "蒸馏对齐增强", "蒸馏与特征对齐增强", "接入 self/global dual distillation 与动态蒸馏权重", ["模块能力", "实验配置"], ["plugins/algorithms/fedcads.py"], ["FedCADS"], ["FedAvg", "FedDyn", "SCAFFOLD", "FedDC", "FedVRA", "FedGKD-P", "A-FedPD", "FedFLD", "FedDF", "FedKD"]),
    buildModule("robust_aggregation", "鲁棒聚合防护", "聚合前鲁棒防御筛选", "支持 VERT、Krum、Median、Trimmed Mean、FLDetector", ["总览", "模块能力", "运行监控"], ["plugins/defenses/vert.py"], ["VERT", "Krum", "Median", "Trimmed Mean", "FLDetector"], ["FedAvg", "FLTrust", "FLAME", "Multi-Krum"]),
    buildModule("trust_screening", "可信客户端筛选", "轻量可信客户端筛选", "历史梯度几何一致性与低开销防御", ["模块能力", "运行监控"], ["code1/defenses.py"], ["FLBeeline"], ["FedAvg", "Krum", "Multi-Krum", "Median", "FLAME", "FLTrust"]),
    buildModule("representation_attack", "特征投毒分析", "特征层投毒攻击展示", "聚焦 prototype poisoning 与表征异常检测", ["模块能力", "实验配置"], ["exps/federated_main.py", "lib/update.py"], ["FeaturePoison"]),
    buildModule("vision_benchmark", "视觉分类基准", "视觉分类基准任务", "支持 CIFAR/MNIST/Tiny-ImageNet 的联邦视觉实验展示", ["总览", "应用场景", "结果对比"], ["plugins/apps/vision"], ["FedAvg", "FedDyn", "FedCADS", "RFL-NLCP", "FedCVC", "FedDTC", "GFed-HSAM"], ["FedProx", "SCAFFOLD", "FedNova", "FedAdam", "FedYogi", "FedBN", "Ditto", "pFedMe", "FedProto", "FedDF", "FedKD", "FedDC", "FedVRA", "FedVARP", "FedGKD-P", "A-FedPD", "FedFLD", "FedSpeed", "FedSMOO", "FedTOGA"]),
    buildModule("reid_generalization", "ReID 域泛化应用", "域泛化 ReID 应用页", "展示 CO-EVO 的 semantic anchor 与 style bank", ["应用场景", "结果对比"], ["plugins/apps/reid/co_evo.py"], ["CO-EVO"], ["FedBN", "Ditto", "pFedMe", "FedProto", "SCAFFOLD", "MOON", "FedProx", "MixStyle", "CrossStyle", "FedReID", "FedPav", "SNR", "DACS", "SSCU"]),
    buildModule("multimodal_3d", "3D 多模态应用", "3D 多模态应用页", "展示 FedULIP base-to-new 与 cross-dataset 结果", ["应用场景", "结果对比"], ["plugins/apps/ulip3d/fedulip.py"], ["FedULIP"], ["FedAvg", "ULIP", "PointCLIP", "FedKgCoOp", "FedVPT", "FedTPG", "FedCoCoOp", "FedMaPLe", "FedCLIP", "FedMVP"]),
    buildModule("experiment_toolkit", "实验配置与结果工具", "配置、日志与结果导出工具", "负责配置预览、实验模板、导出与结果复核", ["总览", "实验配置", "结果对比"], ["configs/", "outputs/", "scripts/"], ["导出", "模板"]),
  ];

  const moduleReferences = Object.fromEntries(
    modules.map((module) => [
      module.id,
      {
        implementation_status: module.implementation_status,
        papers: [
          `${module.name} 相关参考资产 A`,
          `${module.name} 相关参考资产 B`,
        ],
        code_sources: module.backend_plugins,
        frontend_views: module.frontend_views,
        backend_plugins: module.backend_plugins,
      },
    ])
  );

  const algorithms = [
    entry("fedavg", "FedAvg", ["core_engine", "vision_benchmark", "participation_adaptation", "optimization_control", "distillation_alignment", "robust_aggregation", "trust_screening", "multimodal_3d"]),
    entry("fedprox", "FedProx", ["core_engine", "vision_benchmark", "participation_adaptation", "optimization_control", "reid_generalization"]),
    entry("scaffold", "SCAFFOLD", ["core_engine", "vision_benchmark", "participation_adaptation", "optimization_control", "distillation_alignment", "reid_generalization"]),
    entry("feddyn", "FedDyn", ["core_engine", "vision_benchmark", "participation_adaptation", "optimization_control", "distillation_alignment"]),
    entry("fednova", "FedNova", ["core_engine", "vision_benchmark"]),
    entry("fedadam", "FedAdam", ["core_engine", "vision_benchmark"]),
    entry("fedyogi", "FedYogi", ["core_engine", "vision_benchmark"]),
    entry("fedbn", "FedBN", ["core_engine", "vision_benchmark", "reid_generalization"]),
    entry("ditto", "Ditto", ["core_engine", "vision_benchmark", "reid_generalization"]),
    entry("pfedme", "pFedMe", ["core_engine", "vision_benchmark", "reid_generalization"]),
    entry("fedproto", "FedProto", ["core_engine", "vision_benchmark", "reid_generalization"]),
    entry("feddf", "FedDF", ["core_engine", "vision_benchmark", "distillation_alignment"]),
    entry("fedkd", "FedKD", ["core_engine", "vision_benchmark", "distillation_alignment"]),
    entry("rfl_nlcp", "RFL-NLCP", ["participation_adaptation", "vision_benchmark"]),
    entry("fedcvc", "FedCVC", ["participation_adaptation", "vision_benchmark"]),
    entry("feddtc", "FedDTC", ["participation_adaptation", "vision_benchmark"]),
    entry("fedcads", "FedCADS", ["distillation_alignment", "vision_benchmark"]),
    entry("gfed_hsam", "GFed-HSAM", ["participation_adaptation", "optimization_control", "vision_benchmark"]),
    entry("a_fedpd", "A-FedPD", ["participation_adaptation", "optimization_control", "distillation_alignment", "vision_benchmark"]),
    entry("a_fedpdsam", "A-FedPDSAM", ["optimization_control"]),
    entry("fedspeed", "FedSpeed", ["participation_adaptation", "optimization_control", "vision_benchmark"]),
    entry("fedsmoo", "FedSMOO", ["participation_adaptation", "optimization_control", "vision_benchmark"]),
    entry("fedlesam_d", "FedLESAM-D", ["optimization_control"]),
    entry("fedgloss", "FedGLOSS", ["optimization_control"]),
    entry("feddc", "FedDC", ["participation_adaptation", "distillation_alignment", "vision_benchmark"]),
    entry("fedvra", "FedVRA", ["participation_adaptation", "optimization_control", "distillation_alignment", "vision_benchmark"]),
    entry("fedvarp", "FedVARP", ["participation_adaptation", "optimization_control", "vision_benchmark"]),
    entry("fedtoga", "FedTOGA", ["participation_adaptation", "optimization_control", "vision_benchmark"]),
    entry("fedgkd_p", "FedGKD-P", ["distillation_alignment", "vision_benchmark"]),
    entry("fedfld", "FedFLD", ["distillation_alignment", "vision_benchmark"]),
    entry("co_evo", "CO-EVO", ["reid_generalization"]),
    entry("moon", "MOON", ["reid_generalization"]),
    entry("mixstyle", "MixStyle", ["reid_generalization"]),
    entry("crossstyle", "CrossStyle", ["reid_generalization"]),
    entry("fedreid", "FedReID", ["reid_generalization"]),
    entry("fedpav", "FedPav", ["reid_generalization"]),
    entry("snr", "SNR", ["reid_generalization"]),
    entry("dacs", "DACS", ["reid_generalization"]),
    entry("sscu", "SSCU", ["reid_generalization"]),
    entry("fedulip", "FedULIP", ["multimodal_3d"]),
    entry("ulip", "ULIP", ["multimodal_3d"]),
    entry("pointclip", "PointCLIP", ["multimodal_3d"]),
    entry("fedkgcoop", "FedKgCoOp", ["multimodal_3d"]),
    entry("fedvpt", "FedVPT", ["multimodal_3d"]),
    entry("fedtpg", "FedTPG", ["multimodal_3d"]),
    entry("fedcocoop", "FedCoCoOp", ["multimodal_3d"]),
    entry("fedmaple", "FedMaPLe", ["multimodal_3d"]),
    entry("fedclip", "FedCLIP", ["multimodal_3d"]),
    entry("fedmvp", "FedMVP", ["multimodal_3d"]),
    entry("vert", "VERT", ["robust_aggregation"]),
    entry("krum", "Krum", ["robust_aggregation", "trust_screening"]),
    entry("multi_krum", "Multi-Krum", ["robust_aggregation", "trust_screening"]),
    entry("median", "Median", ["robust_aggregation", "trust_screening"]),
    entry("trimmed_mean", "Trimmed Mean", ["robust_aggregation"]),
    entry("fldetector", "FLDetector", ["robust_aggregation"]),
    entry("flbeeline", "FLBeeline", ["trust_screening"]),
    entry("fltrust", "FLTrust", ["robust_aggregation", "trust_screening"]),
    entry("flame", "FLAME", ["robust_aggregation", "trust_screening"]),
    entry("featurepoison", "FeaturePoison", ["representation_attack"]),
  ];

  const datasets = [
    entry("cifar10", "CIFAR-10"),
    entry("cifar100", "CIFAR-100"),
    entry("mnist", "MNIST"),
    entry("tiny_imagenet", "Tiny-ImageNet"),
    entry("fashion_mnist", "FashionMNIST"),
    entry("emnist", "EMNIST"),
    entry("femnist", "FEMNIST"),
    entry("ag_news", "AG News"),
    entry("cuhk02", "CUHK02"),
    entry("cuhk03", "CUHK03"),
    entry("msmt17", "MSMT17"),
    entry("market1501", "Market1501"),
    entry("modelnet40", "ModelNet40"),
    entry("scanobjectnn", "ScanObjectNN"),
    entry("shapenetcore", "ShapeNetCore"),
    entry("mvtec3d", "MVTec3D"),
    entry("mnist3d", "MNIST3D"),
    entry("3dimage", "3Dimage"),
  ];

  const optimizers = [entry("sam", "SAM"), entry("esam", "ESAM"), entry("hsam", "HSAM")];
  const defenses = [
    entry("vert", "VERT"),
    entry("krum", "Krum"),
    entry("median", "Median"),
    entry("trimmed_mean", "Trimmed Mean"),
    entry("fldetector", "FLDetector"),
    entry("flbeeline", "FLBeeline"),
  ];
  const attacks = [
    entry("gn", "GN"),
    entry("mr", "MR"),
    entry("agr", "AGR"),
    entry("alie", "ALIE"),
    entry("feature_poison", "FeaturePoison"),
  ];

  const experiments = [
    {
      id: "exp_demo_001",
      name: "FedCVC 在 CIFAR-10 上的实验",
      algorithm: "fedcvc",
      dataset: "cifar10",
      defense: "vert",
      attack: "feature_poison",
      participation_rate: 0.1,
      seed: 42,
      num_clients: 30,
      rounds: 10,
      current_round: 6,
      status: "running",
      best_accuracy: 0.834,
    },
    {
      id: "exp_demo_006",
      name: "FedDyn 在 Tiny-ImageNet 上的实验",
      algorithm: "feddyn",
      dataset: "tiny_imagenet",
      defense: "none",
      attack: "none",
      participation_rate: 0.15,
      seed: 21,
      num_clients: 30,
      rounds: 10,
      current_round: 5,
      status: "running",
      best_accuracy: 0.791,
    },
    {
      id: "exp_demo_007",
      name: "GFed-HSAM 在 CIFAR-10 上的实验",
      algorithm: "gfed_hsam",
      dataset: "cifar10",
      defense: "flbeeline",
      attack: "gn",
      participation_rate: 0.2,
      seed: 18,
      num_clients: 30,
      rounds: 10,
      current_round: 4,
      status: "running",
      best_accuracy: 0.818,
    },
    {
      id: "exp_demo_008",
      name: "FedULIP 在 ModelNet40 上的实验",
      algorithm: "fedulip",
      dataset: "modelnet40",
      defense: "none",
      attack: "none",
      participation_rate: 0.25,
      seed: 9,
      num_clients: 30,
      rounds: 10,
      current_round: 6,
      status: "running",
      best_accuracy: 0.744,
    },
    {
      id: "exp_demo_002",
      name: "FedCADS 防御演示实验",
      algorithm: "fedcads",
      dataset: "cifar10",
      defense: "vert",
      attack: "agr",
      participation_rate: 0.2,
      seed: 7,
      num_clients: 30,
      rounds: 10,
      current_round: 10,
      status: "completed",
      best_accuracy: 0.852,
    },
    {
      id: "exp_demo_003",
      name: "CO-EVO 在 Market1501 上的实验",
      algorithm: "co_evo",
      dataset: "market1501",
      defense: "none",
      attack: "none",
      participation_rate: 0.3,
      seed: 12,
      num_clients: 30,
      rounds: 10,
      current_round: 10,
      status: "completed",
      best_accuracy: 0.781,
    },
  ];

  const results = [
    result("exp_demo_001", "FedCVC 在 CIFAR-10 上的实验", "fedcvc", "cifar10", "vision_classification", 0.834, 0.823, 0.421, 10, "running", "feature_poison", "vert", 0.1, 42, 0.0, 0.0, 0.88),
    result("exp_demo_002", "FedCADS 防御演示实验", "fedcads", "cifar10", "defense_demo", 0.852, 0.847, 0.316, 10, "completed", "agr", "vert", 0.2, 7, 0.0, 0.0, 0.91),
    result("exp_demo_003", "CO-EVO 在 Market1501 上的实验", "co_evo", "market1501", "reid", 0.781, 0.778, 0.502, 10, "completed", "none", "none", 0.3, 12, 0.781, 0.823, 0.0),
    result("exp_demo_004", "FedULIP 在 ModelNet40 上的实验", "fedulip", "modelnet40", "ulip3d", 0.733, 0.721, 0.587, 10, "completed", "none", "none", 0.2, 5, 0.0, 0.0, 0.0),
    result("exp_demo_005", "FedDyn 在 MNIST 上的实验", "feddyn", "mnist", "vision_classification", 0.942, 0.939, 0.122, 10, "completed", "gn", "flbeeline", 0.15, 99, 0.0, 0.0, 0.86),
  ];

  const compareResults = results.slice(0, 4);

  const templates = [
    { id: "baseline_suite", name: "基础算法基线对比", config: { module_id: "vision_benchmark", task_type: "vision_classification", algorithm: "fedprox", split: "iid", network: "small_cnn", participation_rate: 0.2, num_clients: 30, rounds: 10 } },
    { id: "low_participation", name: "低参与率非IID场景", config: { module_id: "participation_adaptation", task_type: "vision_classification", algorithm: "fedcvc", split: "dirichlet", network: "small_cnn", participation_rate: 0.1, num_clients: 30, rounds: 10 } },
    { id: "poison_defense", name: "投毒攻击与防御对比", config: { module_id: "robust_aggregation", task_type: "defense_demo", split: "dirichlet", network: "small_cnn", defense: "vert", attack: "featurepoison", num_clients: 30, rounds: 10 } },
    { id: "reid_baseline", name: "ReID 域泛化基准", config: { module_id: "reid_generalization", task_type: "reid", dataset: "market1501", split: "domain_as_client", network: "tiny_reid", algorithm: "co_evo", num_clients: 30, rounds: 10 } },
    { id: "fedulip_3d", name: "FedULIP 3D 多模态", config: { module_id: "multimodal_3d", task_type: "ulip3d", dataset: "modelnet40", split: "iid", protocol: "base2new", network: "pointbert", algorithm: "fedulip", participation_rate: 0.5, num_clients: 30, rounds: 10 } },
  ];

  const metrics = {
    exp_demo_001: {
      history: buildHistory(10, 0.71, 0.84),
      participation: [
        { round: 5, participating: [1, 2, 3, 4, 5, 7, 8], malicious: [14] },
        { round: 10, participating: [2, 3, 5, 7, 10, 11, 15], malicious: [14, 18] },
      ],
      defense: [
        { round: 1, trusted_clients: [1, 3, 4, 5, 8, 9, 21, 24, 26], filtered_clients: [2, 6, 10], detection_accuracy: 0.7849 },
        { round: 2, trusted_clients: [1, 7, 8, 14, 18, 21, 23, 27], filtered_clients: [2, 12], detection_accuracy: 0.7739 },
        { round: 3, trusted_clients: [4, 5, 7, 9, 11, 24, 29], filtered_clients: [18], detection_accuracy: 0.7645 },
        { round: 4, trusted_clients: [2, 9, 15, 20, 24, 28], filtered_clients: [16], detection_accuracy: 0.7618 },
        { round: 5, trusted_clients: [7, 12, 19, 20, 21, 25], filtered_clients: [2], detection_accuracy: 0.8492 },
        { round: 6, trusted_clients: [3, 4, 8, 9, 13, 22, 24], filtered_clients: [10, 17], detection_accuracy: 0.7814 },
        { round: 7, trusted_clients: [3, 20, 21, 22, 23, 30], filtered_clients: [9, 19], detection_accuracy: 0.7745 },
        { round: 8, trusted_clients: [2, 8, 9, 11, 18, 21, 27], filtered_clients: [3, 15], detection_accuracy: 0.8102 },
        { round: 9, trusted_clients: [7, 19, 23, 26], filtered_clients: [8, 14, 17], detection_accuracy: 0.8827 },
        { round: 10, trusted_clients: [5, 8, 9, 15, 18, 19, 22, 23, 30], filtered_clients: [3, 4, 7, 17], detection_accuracy: 0.8264 },
      ],
      drift: {
        ema_heterogeneity: 0.742,
        gradient_drift: 0.214,
        compensation: 0.133,
      },
    },
  };

  const eventTimelines = {
    exp_demo_001: [
      { type: "round_complete", payload: { round: 7, train_loss: 0.392, test_accuracy: 0.827, timestamp: "2026-06-24T10:00:01Z" } },
      { type: "client_participation", payload: { round: 8, participating: [2, 4, 6, 8, 10, 12], malicious: [18] } },
      { type: "round_complete", payload: { round: 8, train_loss: 0.386, test_accuracy: 0.829, timestamp: "2026-06-24T10:00:03Z" } },
      { type: "defense_detection", payload: { round: 8, trusted_clients: [2, 4, 6, 8, 10], filtered_clients: [18, 20], detection_accuracy: 0.884 } },
      { type: "round_complete", payload: { round: 9, train_loss: 0.381, test_accuracy: 0.831, timestamp: "2026-06-24T10:00:05Z" } },
      { type: "completed", payload: { final_accuracy: 0.845, final_loss: 0.301, message: "演示任务已正常完成。" } },
    ],
  };

  return {
    overview: {
      running_experiments: 4,
      algorithm_count: algorithms.length,
      dataset_count: datasets.length,
      defense_count: 6,
      key_metrics: {
        accuracy: 0.847,
        map: 0.781,
        rank1: 0.823,
        attack_detection_accuracy: 0.884,
        communication_cost: 128,
      },
    },
    capabilityMap: modules.map((module) => ({
      id: module.id,
      name: module.name,
      positioning: module.positioning,
      tags: module.algorithms.slice(0, 3).map((item) => item.name),
    })),
    modules,
    moduleReferences,
    algorithms,
    datasets,
    optimizers,
    defenses,
    attacks,
    experiments,
    results,
    compareResults,
    templates,
    metrics,
    eventTimelines,
  };
}

function buildModule(id, name, positioning, problemFocus, frontendViews, backendPlugins, algorithmNames, baselineNames = []) {
  const moduleMeta = getModuleMeta(id);
  const primaryIds = uniqueValues(algorithmNames.map(normalizeAlgorithmId));
  const baselineIds = uniqueValues(baselineNames.map(normalizeAlgorithmId));
  const baselineIdSet = new Set(baselineIds);
  const allNames = uniqueValues([...algorithmNames, ...baselineNames]);
  return {
    id,
    name,
    positioning,
    implementation_status: moduleMeta.implementation_status,
    description: `${name} 负责 ${positioning}，模块下可选择多种算法方案来实现相同或相近的功能目标。`,
    problem_focus: problemFocus,
    io_contract: "输入为实验配置、客户端状态与算法参数；输出为模型更新、指标、日志与结果摘要。",
    frontend_views: frontendViews,
    showcase_points: moduleMeta.showcase_points,
    backend_service: moduleMeta.backend_service,
    config_fields: moduleMeta.config_fields,
    backend_plugins: backendPlugins,
    primary_algorithms: primaryIds,
    baseline_algorithms: baselineIds,
    algorithms: allNames.map((algorithmName) => {
      const algorithmId = normalizeAlgorithmId(algorithmName);
      const smokeValidated = SMOKE_VALIDATED_ALGORITHM_IDS.has(algorithmId);
      return {
        id: algorithmId,
        name: algorithmName,
        category: baselineIdSet.has(algorithmId) ? "paper_baseline" : "algorithm",
        status: smokeValidated ? "validated" : baselineIdSet.has(algorithmId) ? "paper_only" : moduleMeta.implementation_status,
      };
    }),
  };
}

function buildHistory(rounds, startAcc, endAcc) {
  return Array.from({ length: rounds }, (_, index) => {
    const round = index + 1;
    return {
      round,
      train_loss: Number((0.92 - round * 0.045).toFixed(3)),
      test_accuracy: Number((startAcc + ((endAcc - startAcc) / rounds) * round).toFixed(3)),
      map: Number((0.62 + round * 0.012).toFixed(3)),
      rank1: Number((0.68 + round * 0.01).toFixed(3)),
      detection_accuracy: Number((0.55 + round * 0.018).toFixed(3)),
      communication_cost_mb: Number((126 + Math.sin(round * 0.7) * 8).toFixed(2)),
      gradient_drift: Number((0.08 * Math.exp(-round / Math.max(rounds, 1)) + 0.006 * Math.sin(round)).toFixed(4)),
      client_participation_rate: Number((0.12 + 0.025 * Math.sin(round * 0.9)).toFixed(4)),
      timestamp: `2026-06-24T10:${String(index).padStart(2, "0")}:00Z`,
    };
  });
}

function result(experimentId, experimentName, algorithm, dataset, taskType, bestAccuracy, finalAccuracy, bestLoss, rounds, status, attack = "none", defense = "none", participationRate = 0.1, seed = 42, map = 0, rank1 = 0, detectionAccuracy = 0) {
  return {
    experiment_id: experimentId,
    experiment_name: experimentName,
    algorithm,
    dataset,
    task_type: taskType,
    attack,
    defense,
    participation_rate: participationRate,
    seed,
    map,
    rank1,
    detection_accuracy: detectionAccuracy,
    best_accuracy: bestAccuracy,
    final_accuracy: finalAccuracy,
    best_loss: bestLoss,
    communication_rounds: rounds,
    status,
  };
}

function entry(id, name, moduleIds = []) {
  return { id, name, module_ids: moduleIds };
}

function getModuleMeta(id) {
  const metaMap = {
    core_engine: {
      implementation_status: "implemented",
      showcase_points: [
        "Server、Client、Trainer、StateStore、Registry 的训练流程图",
        "核心训练循环与状态流转说明",
      ],
      backend_service: "Core Orchestration Service",
      config_fields: ["task_type", "rounds", "local_epochs", "num_clients", "seed"],
    },
    participation_adaptation: {
      implementation_status: "mocked",
      showcase_points: [
        "Non-IID、低参与率、silence table、EMA 异质性、梯度漂移补偿",
        "低参与异构场景下的漂移修正思路",
      ],
      backend_service: "Adaptation Strategy Service",
      config_fields: ["split", "participation_rate", "silence_rate", "learning_rate"],
    },
    optimization_control: {
      implementation_status: "planned",
      showcase_points: [
        "FedCVC、GFed-HSAM、FedDyn、FedProx、SCAFFOLD、SAM/ESAM/HSAM 对比",
        "优化器选择与 sharpness-aware 约束示意",
      ],
      backend_service: "Optimizer Control Service",
      config_fields: ["optimizer", "learning_rate", "batch_size"],
    },
    distillation_alignment: {
      implementation_status: "implemented",
      showcase_points: [
        "self/global dual distillation、动态蒸馏权重、特征对齐",
        "蒸馏分支与表征对齐关系图",
      ],
      backend_service: "Distillation Alignment Service",
      config_fields: ["algorithm", "distill_weight", "global_distill_weight", "temperature"],
    },
    robust_aggregation: {
      implementation_status: "validated",
      showcase_points: [
        "VERT、Krum、Median、Trimmed Mean、FLDetector 和攻击模拟",
        "可信客户端筛选与聚合前防护示意",
      ],
      backend_service: "Robust Defense Service",
      config_fields: ["defense", "attack", "trust_threshold"],
    },
    trust_screening: {
      implementation_status: "mocked",
      showcase_points: [
        "历史梯度几何一致性、可信客户端筛选、低开销防御",
        "Top-K 可信客户端筛选示意",
      ],
      backend_service: "Trust Screening Service",
      config_fields: ["defense", "participation_rate", "trust_threshold"],
    },
    representation_attack: {
      implementation_status: "planned",
      showcase_points: [
        "特征层投毒、原型异常检测、表征安全",
        "prototype poisoning 与异常表征检测示意",
      ],
      backend_service: "Representation Security Service",
      config_fields: ["attack", "defense", "split"],
    },
    vision_benchmark: {
      implementation_status: "implemented",
      showcase_points: [
        "CIFAR/MNIST/Tiny-ImageNet、IID/Dirichlet/Pathological split",
        "分类任务样例与基础算法对比",
      ],
      backend_service: "Vision Benchmark Service",
      config_fields: ["dataset", "split", "algorithm"],
    },
    reid_generalization: {
      implementation_status: "mocked",
      showcase_points: [
        "CO-EVO、semantic anchor、global style bank、mAP/Rank-1",
        "ReID 域泛化机制与指标对比",
      ],
      backend_service: "ReID Generalization Service",
      config_fields: ["dataset", "algorithm", "rounds"],
    },
    multimodal_3d: {
      implementation_status: "planned",
      showcase_points: [
        "FedULIP、ULIP backbone、adapter/SACA、跨数据集评测",
        "base-to-new/cross-dataset/domain ABCD 评测协议",
      ],
      backend_service: "3D Multimodal Service",
      config_fields: ["dataset", "algorithm", "task_type"],
    },
    experiment_toolkit: {
      implementation_status: "implemented",
      showcase_points: [
        "配置、日志、checkpoint、结果表格、批量任务",
        "实验模板、导出与结果复核工具链",
      ],
      backend_service: "Experiment Utility Service",
      config_fields: ["seed", "export_format", "template_id"],
    },
  };
  return metaMap[id] || {
    implementation_status: "planned",
    showcase_points: [],
    backend_service: "Module Service",
    config_fields: [],
  };
}

function normalizeAlgorithmId(name) {
  return name.toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "");
}

function toQueryString(query) {
  const params = new URLSearchParams();
  Object.entries(query).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== "") {
      params.set(key, String(value));
    }
  });
  const str = params.toString();
  return str ? `?${str}` : "";
}

function formatPercent(value) {
  return `${(Number(value || 0) * 100).toFixed(1)}%`;
}

function isDetectionMetricApplicable(item) {
  if (!item) return false;
  return (item.defense && item.defense !== "none") || (item.attack && item.attack !== "none");
}

function formatDetectionAccuracy(value, item) {
  if (!isDetectionMetricApplicable(item)) {
    return "N/A";
  }
  return formatPercent(value);
}

function shorten(text, maxLength) {
  return text.length > maxLength ? `${text.slice(0, maxLength)}…` : text;
}

function synthesizeLossSeries(bestLoss, count = 8) {
  const startLoss = Math.max(bestLoss + 0.28, bestLoss * 1.9, 0.36);
  return Array.from({ length: count }, (_, index) => {
    const progress = index / Math.max(count - 1, 1);
    const curved = Math.pow(progress, 0.82);
    const value = startLoss - (startLoss - bestLoss) * curved;
    return Number(value.toFixed(3));
  });
}

function synthesizeRobustnessSeries(baseScore, count = 8) {
  const start = Math.max(baseScore - 0.12, 0.42);
  const end = Math.min(baseScore + 0.03, 0.98);
  return Array.from({ length: count }, (_, index) => {
    const progress = index / Math.max(count - 1, 1);
    const value = start + (end - start) * Math.pow(progress, 0.9);
    return Number(value.toFixed(3));
  });
}

function escapeHtml(text) {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function escapeClassName(text) {
  return String(text).replace(/[^a-zA-Z0-9_-]/g, "_").toLowerCase();
}

function randomMetric(min, max) {
  return (Math.random() * (max - min) + min).toFixed(3);
}
