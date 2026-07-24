"""직원 후기 semantic 검색 테스트.

검증 대상:
1. 자연어를 자연어로 찾는가 — 부분문자열로는 안 걸리는 질의가 의미로 매칭되는가
2. 리치 필드(직무·재직상태·평점·한줄평)가 검색 결과에 보존되는가
3. 세 소스의 컬럼명 차이가 공통 스키마로 정규화되는가

`rank_review_rows`는 데이터 로딩과 분리된 순수 함수라 parquet 없이 검증한다.
"""
import unittest

import pandas as pd

from src.dynamic.review_store import (
    REVIEW_COLUMNS,
    _normalize_source,
    rank_review_rows,
)


def _rows() -> pd.DataFrame:
    return pd.DataFrame([
        {"corp": "테스트", "stock_code": "000000", "period_value": "2023", "review_date": "2023.03.07",
         "position": "개발자", "status": "현직원", "rating": "4", "summary": "성장하기 좋은 곳",
         "pros": "연봉이 높고 복지가 좋다", "cons": "업무 강도가 세다", "source_file": "b_company_review.csv"},
        {"corp": "테스트", "stock_code": "000000", "period_value": "2023", "review_date": "2023.02.01",
         "position": "기획자", "status": "전직원", "rating": "2", "summary": "커리어 정체",
         "pros": "워라밸은 괜찮음", "cons": "승진 기회가 없고 커리어 향상이 전혀 안 된다", "source_file": "j_company_review.csv"},
        {"corp": "테스트", "stock_code": "000000", "period_value": "2022", "review_date": "2022.11.01",
         "position": "영업", "status": "현직원", "rating": "3", "summary": "평범",
         "pros": "동료들이 좋다", "cons": "급여 인상률이 낮다", "source_file": "employee_reviews.csv"},
    ])


class TfidfSearchTests(unittest.TestCase):
    def test_finds_despite_spacing_and_particle_variation(self) -> None:
        """char n-gram TF-IDF의 실제 강점: 정확 substring이 없어도 매칭한다.

        질의 '커리어 향상 부족'은 후기의 '커리어 향상이 전혀 안 된다'에 substring으로는
        걸리지 않는다(띄어쓰기·조사가 다르다). 문자 n-gram이 이를 이어준다.
        이것은 어휘 유사도이지 임베딩 의미 매칭이 아니다 — '성장'을 '커리어'로
        이해하는 능력은 없고, 표기 변형에 강인한 것이 이 검색의 실제 가치다.
        """
        results = rank_review_rows(_rows(), "커리어 향상 부족", top_k=1)
        self.assertEqual(len(results), 1)
        top = results[0]
        self.assertIn("커리어 향상", top["cons"])
        self.assertEqual(top["match_method"], "tfidf")
        self.assertGreater(top["relevance"], 0)

    def test_beats_substring_which_would_miss(self) -> None:
        """같은 질의를 substring 카운트로 세면 0건이라, TF-IDF가 실질적 향상임을 고정한다."""
        text = "커리어 정체 워라밸은 괜찮음 승진 기회가 없고 커리어 향상이 전혀 안 된다"
        self.assertEqual(text.count("커리어 향상 부족"), 0)  # substring은 못 찾음
        results = rank_review_rows(_rows(), "커리어 향상 부족", top_k=1)
        self.assertIn("커리어 향상", results[0]["cons"])  # TF-IDF는 찾음

    def test_preserves_rich_fields(self) -> None:
        results = rank_review_rows(_rows(), "연봉 복지", top_k=1)
        top = results[0]
        for field in ["position", "status", "rating", "summary", "pros", "cons", "source_file"]:
            self.assertIn(field, top)
        self.assertEqual(top["position"], "개발자")
        self.assertEqual(top["status"], "현직원")

    def test_empty_subset_returns_empty(self) -> None:
        empty = pd.DataFrame(columns=REVIEW_COLUMNS)
        self.assertEqual(rank_review_rows(empty, "무엇이든", top_k=3), [])

    def test_rows_without_text_are_dropped(self) -> None:
        blank = pd.DataFrame([{
            "corp": "x", "stock_code": "000000", "period_value": "2023", "review_date": None,
            "position": None, "status": None, "rating": None, "summary": None,
            "pros": None, "cons": None, "source_file": "employee_reviews.csv",
        }])
        self.assertEqual(rank_review_rows(blank, "질의", top_k=3), [])

    def test_top_k_is_respected(self) -> None:
        results = rank_review_rows(_rows(), "회사 업무", top_k=2)
        self.assertLessEqual(len(results), 2)


class SourceNormalizationTests(unittest.TestCase):
    """세 소스의 서로 다른 컬럼명이 공통 스키마로 수렴하는지."""

    def test_blind_columns_map_to_pros_cons(self) -> None:
        raw = pd.DataFrame([{
            "corp": "비에이치", "stock_code": "90460", "date": "2023.03.07",
            "position": "품질관리", "status": "현직원", "rating": "1", "summary": "한줄평",
            "blind_up": "장점 텍스트", "blind_down": "단점 텍스트",
        }])
        out = _normalize_source(raw, "blind_up", "blind_down", "b_company_review.csv")
        self.assertEqual(list(out.columns), REVIEW_COLUMNS)
        self.assertEqual(out.iloc[0]["pros"], "장점 텍스트")
        self.assertEqual(out.iloc[0]["cons"], "단점 텍스트")
        self.assertEqual(out.iloc[0]["stock_code"], "090460")  # zero-padded
        self.assertEqual(out.iloc[0]["period_value"], "2023")   # date에서 연도 추출

    def test_jobplanet_columns_map_to_pros_cons(self) -> None:
        raw = pd.DataFrame([{
            "corp": "테스트", "stock_code": "000000", "date": "2022.05.01",
            "position": "개발", "status": "전직원", "rating": "3", "summary": "요약",
            "jobp_up": "잡플 장점", "jobp_down": "잡플 단점", "manager": "리더 평",
        }])
        out = _normalize_source(raw, "jobp_up", "jobp_down", "j_company_review.csv")
        self.assertEqual(out.iloc[0]["pros"], "잡플 장점")
        self.assertEqual(out.iloc[0]["period_value"], "2022")

    def test_employee_reviews_without_rich_fields(self) -> None:
        raw = pd.DataFrame([{
            "corp": "SBS", "stock_code": "34120", "year": "2022",
            "up": "장점", "down": "단점",
        }])
        out = _normalize_source(raw, "up", "down", "employee_reviews.csv")
        self.assertEqual(out.iloc[0]["pros"], "장점")
        self.assertIsNone(out.iloc[0]["position"])  # 없는 리치 필드는 None
        self.assertEqual(out.iloc[0]["period_value"], "2022")


if __name__ == "__main__":
    unittest.main()
