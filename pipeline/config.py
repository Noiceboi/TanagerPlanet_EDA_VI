import json
import pathlib
import yaml

_PIPELINE_DIR = pathlib.Path(__file__).parent
INDICES_YAML_PATH = _PIPELINE_DIR / "indices.yaml"


def load_indices_config(path=None):
    p = pathlib.Path(path) if path else INDICES_YAML_PATH
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_datasets_registry(docs_root):
    p = pathlib.Path(docs_root) / "data" / "datasets.json"
    if p.exists():
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return {"datasets": {}, "_meta": {"generated_by": "run_pipeline.py", "schema_version": 1}}


def save_datasets_registry(registry, docs_root):
    p = pathlib.Path(docs_root) / "data" / "datasets.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)
