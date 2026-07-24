"""ML 신용등급 예측 도구.

원래 이 도구는 `financial_features: dict`를 받아 그대로 모델에 넣었다. 즉 LLM이
학습 시점의 피처 이름과 순서를 50개 가까이 정확히 복원해 내야 동작하는 구조였고,
당연히 실사용에서는 한 번도 성공하지 못했다. 모델이 요구하는 것은 자연어 추론이
아니라 **고정된 피처 계약**이므로, 그 계약을 LLM에게 맡길 이유가 없다.

그래서 도구 인터페이스를 뒤집었다.
- 입력: `stock_code`(+선택 `year`) — LLM이 이미 확정해 둔 식별자
- 피처 조립: 학습 테이블(`credit_model_a.csv`)에서 해당 기업 행을 찾아 학습 때와
  동일하게 식별자/레이블 컬럼만 제거해 재현
- 출력: 예측 등급 + 사용한 기준연도 + 피처 출처

학습 스크립트(archive/legacy/feature/credit_prediction.py)가 다음과 같이 학습했고,
그 전처리를 그대로 재현한다:
    X = model_a[model_a["rank"].notna()].drop(["corp","stock_code","sector","year"]).drop("rank")
"""
import os
import pickle
from functools import lru_cache
from typing import Any, Optional

import pandas as pd

from src.config import BASE_DIR, DATA_DIR, CSV_FILES


RANK_MAPPING = {
    "AAA": 9, "AA+": 8, "AA": 7, "AA-": 6,
    "A+": 5, "A": 4, "A-": 3, "BBB": 2, "JB": 1,
}
RANK_MAPPING_REVERSE = {v: k for k, v in RANK_MAPPING.items()}

# 학습 시 X에서 제외된 컬럼. 식별자 4개 + 레이블 1개.
NON_FEATURE_COLUMNS = ["corp", "stock_code", "sector", "year", "rank"]

# 학습 산출물 탐색 경로.
MODEL_SEARCH_PATHS = (
    os.path.join(BASE_DIR, "models", "rf_credit_rating.pkl"),
    os.path.join(BASE_DIR, "rf_model.pkl"),
)


@lru_cache(maxsize=1)
def _load_model():
    """학습된 RandomForest 모델을 로드합니다. 없거나 못 읽으면 None."""
    model_path = next((path for path in MODEL_SEARCH_PATHS if os.path.exists(path)), None)
    if model_path is None:
        return None
    try:
        with open(model_path, "rb") as f:
            return pickle.load(f)
    except Exception:
        # scikit-learn 미설치 또는 버전 불일치. 도구를 죽이지 않고 비활성으로 둔다.
        return None


def _normalize_stock_code(value: Any) -> str | None:
    text = str(value or "").strip()
    digits = "".join(ch for ch in text if ch.isdigit())
    return digits.zfill(6)[-6:] if digits else None


@lru_cache(maxsize=1)
def _training_table() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, CSV_FILES["credit_model_a"])
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "stock_code" in df.columns:
        df["stock_code"] = df["stock_code"].map(_normalize_stock_code)
    return df


def build_feature_row(stock_code: str, year: int | None = None) -> dict[str, Any]:
    """학습 테이블에서 해당 기업의 피처 행을 학습 때와 동일한 형태로 재현합니다."""
    code = _normalize_stock_code(stock_code)
    table = _training_table()
    if not code:
        return {"available": False, "reason": "stock_code를 해석할 수 없습니다."}
    if table.empty:
        return {"available": False, "reason": f"학습 피처 테이블({CSV_FILES['credit_model_a']})이 없습니다."}

    subset = table[table["stock_code"] == code]
    if subset.empty:
        return {"available": False, "reason": f"stock_code {code}가 학습 피처 테이블에 없습니다."}
    if year is not None and "year" in subset.columns:
        matched = subset[subset["year"].astype(str) == str(year)]
        if not matched.empty:
            subset = matched
    if "year" in subset.columns:
        subset = subset.sort_values("year")

    row = subset.iloc[-1]
    features = row.drop(labels=[c for c in NON_FEATURE_COLUMNS if c in subset.columns]).to_dict()
    return {
        "available": True,
        "stock_code": code,
        "year": str(row.get("year")) if "year" in subset.columns else None,
        "features": features,
        "source_file": CSV_FILES["credit_model_a"],
    }


def predict_credit_rating(stock_code: str, year: int | None = None) -> dict[str, Any]:
    """stock_code 기준으로 피처를 조립해 신용등급을 예측합니다."""
    model = _load_model()
    if model is None:
        return {
            "available": False,
            "error": "신용등급 예측 모델을 사용할 수 없습니다. "
                     "models/rf_credit_rating.pkl과 scikit-learn 설치를 확인하세요.",
        }

    assembled = build_feature_row(stock_code, year=year)
    if not assembled.get("available"):
        return {"available": False, "error": assembled.get("reason")}

    features = assembled["features"]
    # 모델이 학습 시 컬럼명을 기억하고 있으면 그 순서를 정본으로 삼는다.
    expected = list(getattr(model, "feature_names_in_", []))
    if expected:
        missing = [name for name in expected if name not in features]
        if missing:
            return {
                "available": False,
                "error": f"피처 계약 불일치. 누락 컬럼 {missing[:5]}"
                         f"{' 외 %d개' % (len(missing) - 5) if len(missing) > 5 else ''}",
            }
        frame = pd.DataFrame([{name: features[name] for name in expected}], columns=expected)
    else:
        frame = pd.DataFrame([features])

    try:
        prediction = model.predict(frame)
    except Exception as e:
        return {"available": False, "error": f"신용등급 예측 실패: {e}"}

    score = int(prediction[0])
    return {
        "available": True,
        "stock_code": assembled["stock_code"],
        "feature_year": assembled["year"],
        "predicted_rating": RANK_MAPPING_REVERSE.get(score, "Unknown"),
        "numeric_score": score,
        "feature_count": len(frame.columns),
        "source_file": assembled["source_file"],
        "note": "실제 평가기관 등급이 아니라 RandomForest 모델의 예측값입니다.",
    }


def get_feature_importance(top_n: int = 10) -> Optional[list]:
    """모델의 피처 중요도 Top-N을 반환합니다."""
    model = _load_model()
    if model is None:
        return None

    try:
        importances = model.feature_importances_
        feature_names = (
            list(model.feature_names_in_)
            if hasattr(model, "feature_names_in_")
            else [f"feature_{i}" for i in range(len(importances))]
        )
        sorted_indices = importances.argsort()[::-1][:top_n]
        return [
            {"feature": feature_names[i], "importance": round(float(importances[i]), 6)}
            for i in sorted_indices
        ]
    except Exception:
        return None
