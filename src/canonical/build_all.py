from __future__ import annotations

from src.canonical.build_companies import build_companies
from src.canonical.build_credit_ratings import build_credit_ratings
from src.canonical.build_metrics import build_metrics
from src.canonical.build_observations import build_observations
from src.canonical.build_reviews import build_reviews
from src.canonical.schema import ensure_dirs


def build_all() -> dict[str, object]:
    ensure_dirs()
    summary: dict[str, object] = {}
    summary["companies"] = build_companies()
    summary["metrics"] = build_metrics()
    summary["observations"] = build_observations()
    summary["credit_ratings"] = build_credit_ratings()
    summary["reviews"] = build_reviews()
    return summary


if __name__ == "__main__":
    build_all()

