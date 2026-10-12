"""Control-specific configuration validation."""


VALID_MODES = {"none", "high_off", "low_off", "random_off", "high_weak"}


def validate_control_config(config):
    required = {"control_selection_mode", "control_fraction", "control_weak_weight"}
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"control config is missing: {missing}")
    mode = str(config["control_selection_mode"])
    if mode not in VALID_MODES:
        raise ValueError(f"invalid control_selection_mode: {mode}")
    if not bool(config.get("old_new_separation", False)):
        raise ValueError("all C0-C4 controls require diagonal separation")
    if str(config.get("separation_covariance")) != "diagonal":
        raise ValueError("control experiments require diagonal covariance")
    if str(config.get("adaptive_margin_mode")) != "drift":
        raise ValueError("C0-C4 baseline must be AM-Drift")
    if str(config.get("similarity_relation_mode")) != "legacy_absolute":
        raise ValueError("control experiments replace only the AM-Drift pair mask")
    fraction = float(config["control_fraction"])
    if not 0.0 < fraction < 0.5:
        raise ValueError("control_fraction must be in (0,0.5)")
    weak = float(config["control_weak_weight"])
    if not 0.0 < weak < 1.0:
        raise ValueError("control_weak_weight must be in (0,1)")
    if bool(config.get("similarity_scc_enabled", False)):
        raise ValueError("similarity-aware SCC is not part of C0-C4")
    if bool(config.get("pic_enabled", False)) or bool(config.get("ahn_enabled", False)):
        raise ValueError("PIC/AHN are not part of C0-C4")
    if float(config.get("ceos_replay_fraction", 0.0)) != 0.0:
        raise ValueError("CEOS is not part of C0-C4")

