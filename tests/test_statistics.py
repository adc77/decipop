from decision_lab.evaluation.report import depth_interval


def test_depth_drop_bootstrap_samples_disjoint_family_pools():
    id_rows = [
        {"family_id": name, "label": label, "prediction": label}
        for name in ("a", "b")
        for label in (0, 1, 2)
    ]
    depth_rows = [
        {"family_id": name, "label": label, "prediction": (label + 1) % 3}
        for name in ("c", "d")
        for label in (0, 1, 2)
    ]
    result = depth_interval(id_rows, depth_rows, 100, 0.95, 17)
    assert result["id_minus_depth_accuracy"] == 1
    assert result["interval"] == [1, 1]
