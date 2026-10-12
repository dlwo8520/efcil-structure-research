"""Validation parity with the E3 runner without importing its executable module."""


def validate_followup_compat(config):
    required = {
        "pic_enabled",
        "ahn_enabled",
        "normalized_diagonal",
        "adaptive_margin_mode",
        "ceos_replay_fraction",
        "diagnostic_grad_interval",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"E3-compatible config is missing: {missing}")
    diagonal = bool(config["old_new_separation"]) and float(config["sep_lambda"]) > 0
    if diagonal and config["separation_covariance"] != "diagonal":
        raise ValueError("only diagonal separation is supported")
    if bool(config["pic_enabled"]) or bool(config["ahn_enabled"]):
        raise ValueError("PIC/AHN are outside this isolated similarity-aware study")
    if float(config["ceos_replay_fraction"]) != 0.0:
        raise ValueError("CEOS is outside this isolated similarity-aware study")
    if str(config["adaptive_margin_mode"]) not in {"none", "drift"}:
        raise ValueError("this study uses only none or drift adaptive margin")
    if str(config["adaptive_margin_mode"]) == "drift" and not diagonal:
        raise ValueError("AM-Drift requires active diagonal separation")
    if bool(config["normalized_diagonal"]):
        raise ValueError("normalized diagonal is not part of this study")
    if int(config["diagnostic_grad_interval"]) < 0:
        raise ValueError("diagnostic_grad_interval must be non-negative")

