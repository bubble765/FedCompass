"""Plugin base classes and protocols."""
import numpy as np
from dataclasses import dataclass, field


@dataclass
class ClientState:
    client_id: int
    model_params: "np.ndarray | None" = None
    control_variate: "np.ndarray | None" = None         # SCAFFOLD c_i
    hist_gradient: "np.ndarray | None" = None            # FedDyn history
    local_update_last: "np.ndarray | None" = None        # FedDC
    global_update_last: "np.ndarray | None" = None       # FedDC
    silence_counter: int = 0                             # FedCVC
    trust_score: float = 1.0                             # VERT/FLBeeline


@dataclass
class RoundResult:
    round_num: int
    train_loss: float
    test_accuracy: float
    client_participants: list[int] = field(default_factory=list)
    silent_clients: list[int] = field(default_factory=list)
    malicious_clients: list[int] = field(default_factory=list)
    trusted_clients: list[int] = field(default_factory=list)
    filtered_clients: list[int] = field(default_factory=list)
    detection_accuracy: float = 0.0
    gradient_drift: float = 0.0
    communication_cost_mb: float = 0.0
    extra: dict = field(default_factory=dict)
