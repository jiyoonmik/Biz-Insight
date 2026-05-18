"""
ML 신용등급 예측 도구
- 기존 credit_prediction.py의 하드코딩 로직을 Tool 함수로 모듈화
- RandomForest 모델(.pkl) 로드 → 피쳐 계산 → 예측
"""
import os
import pickle
import pandas as pd
from typing import Optional
from src.config import DATA_DIR, BASE_DIR


RANK_MAPPING = {
    "AAA": 9, "AA+": 8, "AA": 7, "AA-": 6,
    "A+": 5, "A": 4, "A-": 3, "BBB": 2, "JB": 1,
}
RANK_MAPPING_REVERSE = {v: k for k, v in RANK_MAPPING.items()}


def _load_model():
    """학습된 RandomForest 모델을 로드합니다."""
    model_path = os.path.join(BASE_DIR, "rf_model.pkl")
    if not os.path.exists(model_path):
        # feature 디렉토리에서도 탐색
        alt_path = os.path.join(BASE_DIR, "src", "feature", "rf_model.pkl")
        if os.path.exists(alt_path):
            model_path = alt_path
        else:
            return None
    with open(model_path, "rb") as f:
        return pickle.load(f)


def predict_credit_rating_from_data(financial_features: dict) -> dict:
    """
    재무 피쳐 딕셔너리를 받아 신용등급을 예측합니다.

    Args:
        financial_features: 모델 입력에 필요한 피쳐 딕셔너리

    Returns:
        {"predicted_rating": str, "numeric_score": int} 또는 에러 딕셔너리
    """
    model = _load_model()
    if model is None:
        return {"error": "신용등급 예측 모델(rf_model.pkl)을 찾을 수 없습니다."}

    try:
        predict_df = pd.DataFrame([financial_features])
        prediction = model.predict(predict_df)
        rating = RANK_MAPPING_REVERSE.get(prediction[0], "Unknown")
        return {
            "predicted_rating": rating,
            "numeric_score": int(prediction[0]),
        }
    except Exception as e:
        return {"error": f"신용등급 예측 실패: {str(e)}"}


def get_feature_importance(top_n: int = 10) -> Optional[list]:
    """모델의 피쳐 중요도 Top-N을 반환합니다."""
    model = _load_model()
    if model is None:
        return None

    try:
        importances = model.feature_importances_
        feature_names = model.feature_names_in_ if hasattr(model, "feature_names_in_") else [f"feature_{i}" for i in range(len(importances))]

        sorted_indices = importances.argsort()[::-1][:top_n]
        return [
            {"feature": feature_names[i], "importance": float(importances[i])}
            for i in sorted_indices
        ]
    except Exception:
        return None
