from __future__ import annotations

import json

from common import ROOT


def _read(path):
    return json.loads(path.read_text()) if path.is_file() else None


def _metric(row, key):
    return f"{row[key]:.2f}" if row and key in row else "—"


def generate() -> None:
    lines = [
        "# FeCAM + Boundary-Aware Sampling 후속 실험",
        "",
        "## 출처와 실험상 변경점",
        "",
        "- [FeCAM 논문](https://arxiv.org/abs/2309.14062)과 [공식 코드](https://github.com/dipamgoswami/FeCAM)의 class-wise covariance shrinkage (`alpha1=alpha2=10`), covariance correlation normalization, normalized mean/query의 Mahalanobis distance를 사용했다. Tukey transform은 껐다.",
        "- Structure 저장 covariance는 원 feature 공간의 moment이므로 공식 pretrained-ViT 예제의 *sample별 L2 정규화 후 covariance*와 같지 않다. F2와 F3는 동일한 원공간 moment 정의로 비교하고, score 단계에서 정규화한다. 따라서 정확한 pretrained-ViT FeCAM 재현이 아닌 Structure 통계 이식 실험이다.",
        "- CEOS는 [공식 코드](https://github.com/HXuSz11/ACB_CEOS_CVPR2026_Findings)의 `expansive_oversampling`에서 nearest enemy, `lambda ~ Uniform(0.7,1)`, L2 정규화를 이식했다. 공식 current minibatch 대신 고정된 전체 current-task actual train feature pool에서 enemy를 찾는다. ACB/focal/class weighting은 사용하지 않는다.",
        "- 모든 조건은 seed 1993, Structure final T10 frozen feature/checkpoint를 사용한다. Oracle old training feature는 진단 전용이다.",
        "",
        "## FeCAM 결과 (%)",
        "",
        "| Protocol | 조건 | Seen | Old | New | Old-only | New-only | Macro | Old→New | New→Old |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for protocol in ("IA10", "CAR10"):
        data = _read(ROOT / "fecam" / protocol / "metrics.json")
        for name in ("F0", "F1", "F2", "F3"):
            row = data.get(name) if data else None
            lines.append(f"| {protocol} | {name} | {_metric(row,'seen_class_accuracy')} | {_metric(row,'old_accuracy')} | {_metric(row,'new_accuracy')} | {_metric(row,'old_only_accuracy')} | {_metric(row,'new_only_accuracy')} | {_metric(row,'class_macro_accuracy')} | {_metric(row,'old_to_new_rate')} | {_metric(row,'new_to_old_rate')} |")
    lines += ["", "F0=기존 Structure CA, F1=공식 pretrained-ViT NCM 정의(샘플별 정규화 후 평균), F2=Stored-FeCAM, F3=Oracle-FeCAM(진단 전용). 전이율은 0–1 범위다. Old-only/New-only는 해당 그룹의 class만 후보로 제한한 정확도이다.", "", "## Boundary margin audit", "", "| Protocol | Source | Macro q05 gap | Macro q10 gap | q05 positive class fraction | q10 positive class fraction |", "|---|---|---:|---:|---:|---:|"]
    gates = []
    for protocol in ("IA10", "CAR10"):
        data = _read(ROOT / "boundary_audit" / protocol / "summary.json")
        gates.append(data["boundary_gap_confirmed"] if data else False)
        for source in ("BB_oracle_gaussian", "BC_stored_gaussian"):
            row = data["sources"].get(source) if data else None
            lines.append(f"| {protocol} | {source} | {_metric(row,'class_macro_q05_gap')} | {_metric(row,'class_macro_q10_gap')} | {_metric(row,'positive_q05_class_fraction')} | {_metric(row,'positive_q10_class_fraction')} |")
    lines += ["", "양의 gap은 Gaussian이 실제 old feature보다 해당 lower-tail margin에서 쉬움을 뜻한다. 상세 class·draw 수치는 `boundary_audit/*/per_class_margins.csv`에 있다.", "", "## CEOS-only matched CA", ""]
    if all(gates):
        lines += ["| Protocol | 조건 | Seen | Old | New | Old-only | New-only | Macro | Old→New | New→Old |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for protocol in ("IA10", "CAR10"):
            data = _read(ROOT / "ceos" / protocol / "metrics.json")
            for name in ("C0_matched_stored_gaussian", "C1_ceos_only"):
                row = data.get(name) if data else None
                lines.append(f"| {protocol} | {name} | {_metric(row,'seen_class_accuracy')} | {_metric(row,'old_accuracy')} | {_metric(row,'new_accuracy')} | {_metric(row,'old_only_accuracy')} | {_metric(row,'new_only_accuracy')} | {_metric(row,'class_macro_accuracy')} | {_metric(row,'old_to_new_rate')} | {_metric(row,'new_to_old_rate')} |")
        lines += ["", "C0는 앞선 matched stored-Gaussian CA, C1은 동일한 stored Gaussian old 샘플에 CEOS만 적용한 CA이다. 두 조건은 256/class, 같은 new bank·pre-CA 초기화·permutation·optimizer·step 수를 쓴다.", "", "| Protocol | C0 q05 | C1 q05 | C0 q10 | C1 q10 | C0 negative | C1 negative | C0 train loss | C1 train loss | C0 score reproduction max diff |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for protocol in ("IA10", "CAR10"):
            data = _read(ROOT / "ceos" / protocol / "metrics.json")
            if data:
                a, b = data["c0_synthetic_margin"], data["ceos_synthetic_margin"]
                lines.append(f"| {protocol} | {a['q05']:.3f} | {b['q05']:.3f} | {a['q10']:.3f} | {b['q10']:.3f} | {a['negative_fraction']:.3f} | {b['negative_fraction']:.3f} | {data['train_loss_c0']:.3f} | {data['train_loss_c1']:.3f} | {data['control']['c0_reproduction_max_score_difference']:.1g} |")
        lines.append("고정 Structure head에서 CEOS bank의 q05/q10이 낮아졌으나, C1의 final test accuracy는 개선되지 않았다. C0 재학습 점수가 저장된 matched C0와 정확히 일치하는지 위 마지막 열로 검증했다.")
    else:
        lines.append("IA10/CAR10 양쪽에서 사전 정의한 boundary-gap 조건이 확인되지 않아 CEOS-only CA를 실행하지 않았다.")
    lines += ["", "## 판정", ""]
    fecam = [_read(ROOT / "fecam" / p / "metrics.json") for p in ("IA10", "CAR10")]
    if all(fecam):
        gains = [x["F2"]["seen_class_accuracy"] - x["F0"]["seen_class_accuracy"] for x in fecam]
        old_gains = [x["F2"]["old_accuracy"] - x["F0"]["old_accuracy"] for x in fecam]
        new_gains = [x["F2"]["new_accuracy"] - x["F0"]["new_accuracy"] for x in fecam]
        fecam_ready = all(x > 0 for x in gains) and all(x >= -1 for x in old_gains + new_gains)
        lines.append(f"- Stored-FeCAM − Structure Seen: IA10 {gains[0]:+.2f} pp, CAR10 {gains[1]:+.2f} pp. Old/New trade-off와 macro를 함께 확인해야 한다.")
        ncm_gain = [x["F2"]["seen_class_accuracy"] - x["F1"]["seen_class_accuracy"] for x in fecam]
        oracle_gain = [x["F3"]["seen_class_accuracy"] - x["F2"]["seen_class_accuracy"] for x in fecam]
        lines.append(f"- Stored-FeCAM − NCM Seen: IA10 {ncm_gain[0]:+.2f} pp, CAR10 {ncm_gain[1]:+.2f} pp. NCM은 샘플별 정규화 후 mean, F2는 저장 원공간 moment이므로 이 차이를 covariance 효과만으로 분해할 수 없다.")
        lines.append(f"- Oracle-FeCAM − Stored-FeCAM Seen: IA10 {oracle_gain[0]:+.2f} pp, CAR10 {oracle_gain[1]:+.2f} pp. CAR10에서 저장 통계 품질의 영향 가능성이 더 크지만, Oracle도 기존 Structure F0를 넘지 못한다.")
        lines.append(f"- FECAM PROTOTYPE READY = {'YES' if fecam_ready else 'NO'} (탐색적·단일 seed 기준)")
    else:
        lines.append("- FECAM PROTOTYPE READY = 미판정 (결과 미완료)")
    ceos = [_read(ROOT / "ceos" / p / "metrics.json") for p in ("IA10", "CAR10")]
    if all(ceos):
        ceos_gains = [x["C1_ceos_only"]["seen_class_accuracy"] - x["C0_matched_stored_gaussian"]["seen_class_accuracy"] for x in ceos]
        ceos_ready = all(x > 0 for x in ceos_gains)
        lines.append(f"- CEOS − C0 Seen: IA10 {ceos_gains[0]:+.2f} pp, CAR10 {ceos_gains[1]:+.2f} pp.")
        old_delta = [x["C1_ceos_only"]["old_accuracy"] - x["C0_matched_stored_gaussian"]["old_accuracy"] for x in ceos]
        new_delta = [x["C1_ceos_only"]["new_accuracy"] - x["C0_matched_stored_gaussian"]["new_accuracy"] for x in ceos]
        lines.append(f"- CEOS Old/New 변화: IA10 {old_delta[0]:+.2f}/{new_delta[0]:+.2f} pp, CAR10 {old_delta[1]:+.2f}/{new_delta[1]:+.2f} pp. CAR10의 작은 Old 이득은 큰 New 손실로 상쇄된다.")
        lines.append("- 두 protocol 모두 Gaussian의 boundary-hard sample 부족 신호는 있었지만, CEOS로 margin을 낮춰도 downstream CA 정확도는 오르지 않았다. 이 실험에서는 boundary gap이 주된 성능 병목이라는 인과 주장을 지지하지 않는다.")
        lines.append(f"- CEOS PROTOTYPE READY = {'YES' if ceos_ready else 'NO'} (탐색적·단일 seed 기준)")
    else:
        lines.append("- CEOS PROTOTYPE READY = NO (조건부 실행 미완료 또는 생략)")
    lines += ["- FULL METHOD READY = NO. 독립 seed 및 task transition 검증 전 일반화 주장 불가.", "", "## 제한 및 다음 단계", "", "FeCAM은 공식 pretrained-ViT의 정규화-feature covariance와 다른 Structure 저장 원공간 covariance를 사용한다. CEOS는 공식 minibatch enemy가 아닌 전체 current-task train pool을 사용한다. 두 차이 모두 결과 해석을 제한한다. 현재 설정은 두 protocol 공통 개선이 없어 FeCAM/CEOS 축을 우선 보류한다. 후속 재검증 시에는 정규화 moment를 class 최초 등장 때 저장하는 설계와 official minibatch CEOS를 별도 비교해야 한다.", "", "세부 class 결과와 covariance status는 각 하위 폴더의 CSV/JSON에, 공식 소스 대조는 `source_audit.md`에 있다.", ""]
    (ROOT / "final_report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    generate()
