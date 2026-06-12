from __future__ import annotations

from src.dynamic.macro_store import build_macro_store
from src.dynamic.review_store import build_review_store
from src.dynamic.stock_store import build_stock_store


def refresh_dynamic_data() -> None:
    build_stock_store()
    build_review_store()
    build_macro_store()


if __name__ == "__main__":
    refresh_dynamic_data()

