import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "site/assets"


def source_path(path):
    parts = Path(path).parts
    if parts[0] == "runs":
        version = parts[1].removeprefix("study-")
        name = "-".join(parts[2:])
        return ROOT / "reports" / f"{version}-evidence" / name
    return ROOT / path


def read(path):
    return json.loads(source_path(path).read_text())


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    paths = []
    studies = {}
    for version, seeds in (("v7", (731, 947)), ("v8", (1601, 1801))):
        studies[version] = []
        for rep, seed in zip(("r1", "r2"), seeds, strict=True):
            folder = f"runs/study-{version}/{rep}"
            raw = read(f"{folder}/independent-raw-counts.json")
            analysis = read(f"{folder}/analysis.json")
            selection = read(f"{folder}/selection-freeze.json")
            if not raw["valid"]:
                raise ValueError("Unverified raw counts")
            studies[version].append(
                dict(seed=seed, repetition=rep, counts=raw["counts"], primary=analysis["primary"])
            )
            for name in ("independent-raw-counts", "analysis", "selection-freeze"):
                paths.append(f"{folder}/{name}.json")
            choices = selection["choices"]
            studies[version][-1]["selected_updates"] = {
                arm: selection["candidates"][choices[arm]["calibrated_nll"]]["step"]
                for arm in ("control", "treatment")
            }
    paths += ["runs/study-v8/preselected-examples.json", "examples/recorded-support-removal.json"]
    inventory = {}
    for name in paths:
        source = source_path(name)
        inventory[str(source.relative_to(ROOT))] = hashlib.sha256(source.read_bytes()).hexdigest()
    payload = dict(
        studies=studies,
        examples={"r2": read("examples/recorded-support-removal.json")},
        verification=dict(prediction_rows=165344, test_metric_sets=44, cpu_tests=126),
        sources=inventory,
        provenance="Recorded study outputs; no live model inference or new experiments.",
    )
    (DEST / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"evidence_files": len(inventory), "data": "assets/results.json"}))


if __name__ == "__main__":
    main()
