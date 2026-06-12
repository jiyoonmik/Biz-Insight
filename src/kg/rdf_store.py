from __future__ import annotations

from pathlib import Path

from src.canonical.schema import DATA_DIR, PROJECT_ROOT


KG_DIR = DATA_DIR / "kg"
INSTANCE_TTL = KG_DIR / "bizinsight-instances.ttl"
ONTOLOGY_TTL = PROJECT_ROOT / "ontology" / "bizinsight-core.ttl"


def ensure_kg_dir() -> None:
    KG_DIR.mkdir(parents=True, exist_ok=True)


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path

