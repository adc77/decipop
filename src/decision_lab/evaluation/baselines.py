import json

import joblib
import numpy as np
import torch

from decision_lab.baseline import make_lexical_estimator
from decision_lab.data.schema import Example
from decision_lab.data.validate import read_examples
from decision_lab.evaluation.metrics import abstention_thresholds, fit_temperature
from decision_lab.evaluation.study import scored, store_predictions
from decision_lab.training.checkpoint import file_hash


def prepare_baselines(dataset, output, class_labels, config):
    if output.exists():
        raise FileExistsError("Baseline models are already frozen")
    output.mkdir(parents=True)
    train = read_examples(dataset / "train.jsonl")
    calibration = read_examples(dataset / "calibration.jsonl")
    model = make_lexical_estimator(config.seed)
    model.fit([row.prompt for row in train], [row.label.value for row in train])
    permutation = [model.classes_.tolist().index(label) for label in class_labels]
    logits = torch.from_numpy(
        model.decision_function([row.prompt for row in calibration])[:, permutation]
    )
    labels = [class_labels.index(row.label.value) for row in calibration]
    fit = fit_temperature(logits, labels, config)
    fit["thresholds"] = abstention_thresholds(logits, fit["temperature"], config.coverage_targets)
    counts = np.array([sum(row.label.value == label for row in train) for label in class_labels])
    priors = counts / counts.sum()
    majority_logits = torch.from_numpy(np.tile(np.log(priors), (len(calibration), 1)))
    majority = fit_temperature(majority_logits, labels, config)
    majority["thresholds"] = abstention_thresholds(
        majority_logits, majority["temperature"], config.coverage_targets
    )
    joblib.dump(model, output / "lexical.joblib")
    frozen = {
        "train_file_sha256": file_hash(dataset / "train.jsonl"),
        "calibration_file_sha256": file_hash(dataset / "calibration.jsonl"),
        "lexical_model_sha256": file_hash(output / "lexical.joblib"),
        "class_labels": class_labels,
        "class_permutation": permutation,
        "class_priors": priors.tolist(),
        "lexical_calibration": fit,
        "majority_calibration": majority,
        "lexical_configuration": {
            "ngram_range": [1, 1],
            "C": 1.0,
            "max_iter": 1000,
            "solver": "lbfgs",
            "seed": config.seed,
        },
        "majority_convention": "empirical training priors; ties select first class TRUE",
        "test_predictions_inspected": False,
    }
    (output / "freeze.json").write_text(json.dumps(frozen, indent=2) + "\n")
    return frozen


def evaluate_baselines(dataset, directory, test_inputs, config):
    global_freeze = json.loads((directory.parent / "calibration-freeze.json").read_text())
    if file_hash(directory / "freeze.json") != global_freeze["baseline_freeze_sha256"]:
        raise ValueError("Baseline configuration changed after global calibration freeze")
    frozen = json.loads((directory / "freeze.json").read_text())
    if file_hash(directory / "lexical.joblib") != frozen["lexical_model_sha256"]:
        raise ValueError("Lexical baseline weights changed after freeze")
    if file_hash(dataset / "train.jsonl") != frozen["train_file_sha256"]:
        raise ValueError("Baseline training data changed")
    if file_hash(dataset / "calibration.jsonl") != frozen["calibration_file_sha256"]:
        raise ValueError("Baseline calibration data changed")
    model = joblib.load(directory / "lexical.joblib")
    permutation = frozen["class_permutation"]
    class_labels = frozen["class_labels"]
    manifest = json.loads((test_inputs / "manifest.json").read_text())
    splits = {
        name: read_examples(dataset / f"{name}.jsonl")
        for name in ("test_id", "test_depth", "test_wording", "test_structure")
    }
    variants = {}
    for name in manifest["transformations"]:
        payload = [
            json.loads(line) for line in (test_inputs / f"{name}.jsonl").read_text().splitlines()
        ]
        variants["robust_" + name] = payload
        splits["robust_" + name] = [
            Example.model_validate_json(json.dumps(item["example"])) for item in payload
        ]
    result = {"models": {name: {"splits": {}} for name in ("majority", "lexical")}}
    for split, rows in splits.items():
        labels = torch.tensor([class_labels.index(row.label.value) for row in rows])
        lexical_logits = torch.from_numpy(
            model.decision_function([row.prompt for row in rows])[:, permutation]
        )
        majority_logits = torch.from_numpy(np.tile(np.log(frozen["class_priors"]), (len(rows), 1)))
        for name, logits in (("lexical", lexical_logits), ("majority", majority_logits)):
            calibration = frozen[name + "_calibration"]
            predictions = {"logits": logits, "labels": labels}
            result["models"][name]["splits"][split] = scored(predictions, calibration, config)
            store_predictions(
                directory.parent / "predictions" / name / f"{split}.jsonl",
                rows,
                predictions,
                calibration,
                variants[split] if split in variants else None,
            )
    (directory / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
