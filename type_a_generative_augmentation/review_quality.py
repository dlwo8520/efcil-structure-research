"""Human labels for the fixed quality sample. View contact sheets before answering."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from type_a_generative_augmentation.common import ROOT, read_csv, write_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviewer", required=True)
    args = parser.parse_args()
    rows = read_csv(ROOT / "quality_review.csv")
    print("Open contact_sheets. a=identity verified + meaningful variation; d=verified near duplicate; "
          "r=label/object damaged; u=uncertain/unverifiable; q=save and quit.")
    for row in rows:
        if not int(row["review_sample"]) or row["review_status"] != "pending":
            continue
        print(row["record_id"], "class", row["y"], row.get("class_name", ""), "original ID", row.get("original_class_id", ""))
        print("parent:", row["parent_path"], "generated:", row["generated_path"])
        answer = input("Human review [a/d/r/u/q]: ").strip().lower()
        while answer not in ("a", "d", "r", "u", "q"):
            answer = input("Use a/d/r/u/q: ").strip().lower()
        if answer == "q":
            break
        row.update({"review_status": "accept" if answer in ("a", "d") else ("reject" if answer == "r" else "uncertain"),
                    "label_fidelity_verified": "yes" if answer in ("a", "d") else "no",
                    "variation": "meaningful" if answer == "a" else ("near_duplicate" if answer == "d" else "unknown"),
                    "reviewer": args.reviewer, "reviewed_at_utc": datetime.now(timezone.utc).isoformat()})
        if answer in ("r", "u"):
            row["reason"] = input("Reason: ").strip()
        write_csv(ROOT / "quality_review.csv", rows)
    write_csv(ROOT / "quality_review.csv", rows)


if __name__ == "__main__":
    main()
