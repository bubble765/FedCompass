"""Plugin registry: maps algorithm/defense/attack/optimizer IDs to implementations."""
from app.plugins.algorithms import (
    baselines,
    co_evo,
    fedavg,
    fedcads,
    fedcvc,
    feddtc,
    feddyn,
    fedprox,
    fedulip,
    gfed_hsam,
    rfl_nlcp,
    scaffold,
)
from app.plugins.defenses import flame, fldetector, flbeeline, fltrust, krum, median, multi_krum, trimmed_mean, vert
from app.plugins.attacks import (
    agr_byzantine,
    alie,
    featurepoison,
    gaussian_noise,
    model_replacement,
    prototype_attack,
)


def _algorithm(method: str, profile: str = "vision"):
    return baselines.build_algorithm(method, profile)


def _local(method: str, profile: str = "vision"):
    return baselines.make_local_train(method, profile)


ALGORITHM_REGISTRY = {
    # Core implemented baselines
    "fedavg": (fedavg.fedavg_aggregate, fedavg.fedavg_local_train),
    "fedprox": (fedprox.fedprox_aggregate, fedprox.fedprox_local_train),
    "feddyn": (feddyn.feddyn_aggregate, feddyn.feddyn_local_train),
    "scaffold": (scaffold.scaffold_aggregate, scaffold.scaffold_local_train),
    "fednova": _algorithm("fednova"),
    "fedadam": _algorithm("fedadam"),
    "fedyogi": _algorithm("fedyogi"),
    "fedbn": _algorithm("fedbn"),
    "ditto": _algorithm("ditto"),
    "pfedme": _algorithm("pfedme"),
    "fedproto": _algorithm("fedproto"),
    "feddf": _algorithm("feddf"),
    "fedkd": _algorithm("fedkd"),

    # Lab methods
    "fedcvc": (fedcvc.fedcvc_aggregate, fedcvc.fedcvc_local_train),
    "feddtc": (feddtc.feddtc_aggregate, feddtc.feddtc_local_train),
    "rfl_nlcp": (rfl_nlcp.rfl_nlcp_aggregate, rfl_nlcp.rfl_nlcp_local_train),
    "gfed_hsam": (gfed_hsam.gfed_hsam_aggregate, gfed_hsam.gfed_hsam_local_train),
    "fedcads": (fedcads.fedcads_aggregate, fedcads.fedcads_local_train),
    "co_evo": (co_evo.co_evo_aggregate, co_evo.co_evo_local_train),
    "fedulip": (fedulip.fedulip_aggregate, fedulip.fedulip_local_train),

    # Optimizers and paper baselines exposed as runnable algorithm choices
    "sgd": _algorithm("sgd"),
    "sam": _algorithm("sam"),
    "esam": _algorithm("esam"),
    "hsam": _algorithm("hsam"),
    "a_fedpd": _algorithm("a_fedpd"),
    "a_fedpdsam": _algorithm("a_fedpdsam"),
    "fedspeed": _algorithm("fedspeed"),
    "fedsmoo": _algorithm("fedsmoo"),
    "fedlesam_d": _algorithm("fedlesam_d"),
    "fedgloss": _algorithm("fedgloss"),
    "feddc": _algorithm("feddc"),
    "fedvra": _algorithm("fedvra"),
    "fedvarp": _algorithm("fedvarp"),
    "fedtoga": _algorithm("fedtoga"),
    "fedgkd_p": _algorithm("fedgkd_p"),
    "fedfld": _algorithm("fedfld"),

    # Defense methods can also be selected in the module algorithm dropdown
    "vert": (vert.vert_aggregate, _local("vert", "defense")),
    "krum": (krum.krum_aggregate, _local("krum", "defense")),
    "multi_krum": (multi_krum.multi_krum_aggregate, _local("multi_krum", "defense")),
    "median": (median.median_aggregate, _local("median", "defense")),
    "trimmed_mean": (trimmed_mean.trimmed_mean_aggregate, _local("trimmed_mean", "defense")),
    "fldetector": (fldetector.fldetector_aggregate, _local("fldetector", "defense")),
    "flbeeline": (flbeeline.flbeeline_aggregate, _local("flbeeline", "defense")),
    "fltrust": (fltrust.fltrust_aggregate, _local("fltrust", "defense")),
    "flame": (flame.flame_aggregate, _local("flame", "defense")),

    # Attack entries are runnable as stress-test algorithms as well as attacks
    "featurepoison": _algorithm("featurepoison"),
    "fedproto_prototype_attack": _algorithm("fedproto_prototype_attack"),
    "gn": _algorithm("gn"),
    "mr": _algorithm("mr"),
    "agr": _algorithm("agr"),
    "alie": _algorithm("alie"),

    # ReID baselines
    "moon": _algorithm("moon", "reid"),
    "mixstyle": _algorithm("mixstyle", "reid"),
    "crossstyle": _algorithm("crossstyle", "reid"),
    "fedreid": _algorithm("fedreid", "reid"),
    "fedpav": _algorithm("fedpav", "reid"),
    "snr": _algorithm("snr", "reid"),
    "dacs": _algorithm("dacs", "reid"),
    "sscu": _algorithm("sscu", "reid"),

    # 3D multimodal baselines
    "ulip": _algorithm("ulip", "ulip3d"),
    "pointclip": _algorithm("pointclip", "ulip3d"),
    "fedkgcoop": _algorithm("fedkgcoop", "ulip3d"),
    "fedvpt": _algorithm("fedvpt", "ulip3d"),
    "fedtpg": _algorithm("fedtpg", "ulip3d"),
    "fedcocoop": _algorithm("fedcocoop", "ulip3d"),
    "fedmaple": _algorithm("fedmaple", "ulip3d"),
    "fedclip": _algorithm("fedclip", "ulip3d"),
    "fedmvp": _algorithm("fedmvp", "ulip3d"),
}


DEFENSE_REGISTRY = {
    "krum": (krum.krum_aggregate, None),
    "multi_krum": (multi_krum.multi_krum_aggregate, multi_krum.multi_krum_filter),
    "median": (median.median_aggregate, None),
    "trimmed_mean": (trimmed_mean.trimmed_mean_aggregate, None),
    "vert": (vert.vert_aggregate, vert.vert_filter),
    "flbeeline": (flbeeline.flbeeline_aggregate, flbeeline.flbeeline_filter),
    "fldetector": (fldetector.fldetector_aggregate, fldetector.fldetector_filter),
    "fltrust": (fltrust.fltrust_aggregate, fltrust.fltrust_filter),
    "flame": (flame.flame_aggregate, flame.flame_filter),
}


ATTACK_REGISTRY = {
    "featurepoison": featurepoison,
    "feature_poison": featurepoison,
    "fedproto_prototype_attack": prototype_attack,
    "gn": gaussian_noise,
    "mr": model_replacement,
    "agr": agr_byzantine,
    "alie": alie,
}


def get_algorithm(alg_id):
    """Get (aggregate_fn, local_train_fn) for an algorithm, falls back to FedAvg."""
    return ALGORITHM_REGISTRY.get(alg_id, ALGORITHM_REGISTRY["fedavg"])


def get_defense(defense_id):
    """Get (aggregate_fn, filter_fn) for a defense, falls back to FedAvg aggregation."""
    return DEFENSE_REGISTRY.get(defense_id, (fedavg.fedavg_aggregate, None))


def get_attack(attack_id):
    """Get attack transform function, or None."""
    return ATTACK_REGISTRY.get(attack_id)
