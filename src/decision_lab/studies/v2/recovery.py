import json
import shutil
import uuid
from pathlib import Path

from decision_lab.training.checkpoint import file_hash


def preserve_orphan_selections(arm_root: Path, parent, identity):
    head = arm_root / "head"
    durable_step = 0
    if parent is not None:
        checkpoint = Path(parent)
        if not checkpoint.resolve().is_relative_to(head.resolve()):
            raise ValueError("Recovery parent is outside the arm")
        marker = json.loads((checkpoint / "complete.json").read_text())
        if marker["identity"] != identity:
            raise ValueError("Recovery parent has another protocol identity")
        if file_hash(checkpoint / "progress.json") != marker["files"]["progress.json"]:
            raise ValueError("Recovery parent progress checksum mismatch")
        durable_step = json.loads((checkpoint / "progress.json").read_text())["step"]
    preserved = []
    for selection in sorted(head.glob("selected-*")):
        step = int(selection.name.removeprefix("selected-"))
        if step <= durable_step:
            continue
        destination = arm_root / "interrupted-selections" / uuid.uuid4().hex / selection.name
        destination.parent.mkdir(parents=True)
        shutil.move(selection, destination)
        preserved.append({"original": str(selection), "preserved": str(destination)})
    return preserved
