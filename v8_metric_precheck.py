import json

from decision_lab.publication import atomic_json
from decision_lab.training.checkpoint import file_hash
from decision_lab.v3.protocol import check_seal
from decision_lab.v3.study import metrics
from v6_metric_audit import verify_metrics
from v8_lab.analysis import predictions
from v8_lab.data import TESTS
from v8_lab.protocol import ROOT, verify, verify_study


def main():
    destination = ROOT / "metric-precheck.json"
    if destination.exists():
        raise FileExistsError(destination)
    study = verify_study()
    checks = {}
    for rep in ("r1", "r2"):
        root = ROOT / rep
        spec = verify(root)["settings"]
        selected = check_seal(json.loads((root / "selection-freeze.json").read_text()))
        evaluation = json.loads((root / "evaluation.json").read_text())
        if evaluation["selection_identity"] != selected["identity"]:
            raise ValueError("Evaluation selection identity differs")
        for arm in ("control", "treatment"):
            candidate = selected["choices"][arm]["calibrated_nll"]
            temperature = selected["calibrations"][candidate]["temperature"]
            for split in TESTS:
                rows, logits, labels = predictions(root, arm, split)
                path = root / "predictions" / arm / f"{split}.jsonl"
                digest = file_hash(path)
                if digest != evaluation["arms"][arm]["prediction_files"][split]:
                    raise ValueError("Frozen test logits differ")
                rebuilt = metrics(rows, {"logits": logits, "labels": labels}, temperature, spec)
                name = f"{rep}/{arm}/{split}"
                checks[name] = {
                    "prediction_sha256": digest,
                    "comparison": verify_metrics(
                        rebuilt,
                        evaluation["arms"][arm]["splits"][split],
                        logits,
                        labels,
                        temperature,
                    ),
                }
                print(json.dumps({"metric_precheck": name, "valid": True}), flush=True)
    atomic_json(
        destination,
        {
            "valid": True,
            "study_identity": study["identity"],
            "metric_sets_checked": len(checks),
            "all_original_worker_metrics_preserved": True,
            "checks": checks,
            "candidate_retention_or_release_verification_claimed": False,
        },
    )


if __name__ == "__main__":
    main()
