"""Run frozen-feature FeCAM, boundary audit, then conditionally CEOS-only CA."""
from __future__ import annotations

import argparse
import os


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", required=True, help="Physical GPU index, e.g. 4")
    parser.add_argument("--seed", type=int, default=1993)
    parser.add_argument("--stage", choices=("all", "fecam", "boundary", "ceos", "report"), default="all")
    parser.add_argument("--protocols", nargs="+", choices=("IA10", "CAR10"), default=("IA10", "CAR10"))
    args = parser.parse_args()
    if "CUDA_VISIBLE_DEVICES" in os.environ and os.environ["CUDA_VISIBLE_DEVICES"] != args.gpu:
        raise RuntimeError("CUDA_VISIBLE_DEVICES conflicts with --gpu")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    import torch
    from common import ROOT
    from report import generate

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    device = torch.device("cuda:0")
    print(f"physical GPU {args.gpu}; process-local cuda:0; seed={args.seed}", flush=True)
    from fecam.run import run as run_fecam
    from boundary_audit.run import run as run_boundary
    from ceos.run import run as run_ceos

    if args.stage in ("all", "fecam"):
        for protocol in args.protocols:
            if not (ROOT / "fecam" / protocol / "metrics.json").is_file():
                run_fecam(protocol, device, args.seed)
            else:
                print(f"[{protocol} FeCAM] existing result preserved", flush=True)
    if args.stage in ("all", "boundary"):
        for protocol in args.protocols:
            if not (ROOT / "boundary_audit" / protocol / "summary.json").is_file():
                run_boundary(protocol, device, args.seed)
            else:
                print(f"[{protocol} boundary] existing result preserved", flush=True)
    if args.stage in ("all", "ceos"):
        import json
        summaries = {}
        for protocol in ("IA10", "CAR10"):
            path = ROOT / "boundary_audit" / protocol / "summary.json"
            if not path.is_file():
                raise RuntimeError(f"CEOS gate requires both IA10/CAR10 boundary audits: {path}")
            summaries[protocol] = json.loads(path.read_text())
        if all(row["boundary_gap_confirmed"] for row in summaries.values()):
            for protocol in args.protocols:
                if not (ROOT / "ceos" / protocol / "metrics.json").is_file():
                    run_ceos(protocol, device, args.seed)
                else:
                    print(f"[{protocol} CEOS] existing result preserved", flush=True)
        else:
            print("CEOS skipped: boundary gap not confirmed in both protocols", flush=True)
    generate()


if __name__ == "__main__":
    main()
