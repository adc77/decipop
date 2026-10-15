import json

from decision_lab.publication import atomic_json
from decision_lab.studies.v6.study import package_candidate
from decision_lab.training.checkpoint import file_hash, verify_checkpoint


def prepare_resume(root, arm, original_identity, active_identity, step):
    state_path = root / f"state-{arm}.json"
    if state_path.exists():
        raise FileExistsError("Recovery requires a missing progress pointer")
    checkpoint = root / "recovery" / arm / f"step-{step}"
    verify_checkpoint(checkpoint, original_identity)
    progress = json.loads((checkpoint / "progress.json").read_text())
    if (
        progress["arm"] != arm
        or progress["step"] != step
        or len(progress["updates"]) != step
        or progress["complete"]
    ):
        raise ValueError("Recovery checkpoint progress differs")
    partial = root / "candidates" / f"{arm}-step{step}"
    discarded = root / "failed-candidate-packaging" / partial.name
    partial_inventory = {}
    if partial.exists():
        if (root / "candidate-records" / f"{partial.name}.json").exists() or discarded.exists():
            raise ValueError("Refusing to replace a completed or previously retained candidate")
        partial_inventory = {
            str(p.relative_to(partial)): file_hash(p) for p in partial.rglob("*") if p.is_file()
        }
        discarded.parent.mkdir(parents=True, exist_ok=True)
        partial.rename(discarded)
    package_candidate(root, arm, step, progress["presentations"], checkpoint)
    state = {k: v for k, v in progress.items() if k != "updates"} | {
        "resume": str(checkpoint.relative_to(root)),
        "protocol_identity": active_identity,
        "checkpoint_protocol_identity": original_identity,
    }
    atomic_json(state_path, state)
    atomic_json(root / f"trajectory-{arm}.json", progress)
    result = {
        "source_protocol_identity": original_identity,
        "active_protocol_identity": active_identity,
        "arm": arm,
        "step": step,
        "presentations": progress["presentations"],
        "checkpoint_marker": json.loads((checkpoint / "complete.json").read_text()),
        "retained_partial_candidate_files": partial_inventory,
        "partial_candidate_preserved_at": str(discarded.relative_to(root)),
        "checkpoint_hashes_verified": True,
        "optimizer_and_rng_restoration_pending": True,
    }
    atomic_json(root / "checkpoint-recovery.json", result)
    return result
