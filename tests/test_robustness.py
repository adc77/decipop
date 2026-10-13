from pathlib import Path

from decision_lab.config import load_dataset_config
from decision_lab.data.generate import generate_dataset
from decision_lab.data.solve import solve
from decision_lab.data.validate import read_examples
from decision_lab.evaluation.robustness import generate_variants


def test_reserved_variants_preserve_or_recompute_labels_with_the_oracle(tmp_path):
    generate_dataset(load_dataset_config(Path("configs/smoke.yaml")), tmp_path / "data")
    rows = read_examples(tmp_path / "data/test_id.jsonl")
    variants, manifest = generate_variants(rows, 2, 17)
    assert len(variants) == 6
    assert all(value["families"] == 2 for value in manifest.values())
    for transformation, group in variants.items():
        assert len(group) == 6
        for item in group:
            row = item["example"]
            assert solve(row.theory).answer(row.query) == (row.label, row.proof_depth)
            if transformation not in {"remove_premise", "flip_fact"}:
                assert not item["label_changed"]
        if transformation in {"remove_premise", "flip_fact"}:
            assert any(item["label_changed"] for item in group)
