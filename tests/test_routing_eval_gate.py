"""라우팅 평가를 테스트 스위트에 게이트로 묶는다.

평가를 별도 스크립트로만 두면 아무도 돌리지 않아 서서히 낡는다. LLM이 필요 없고
1초 안에 끝나므로 테스트에 포함해 매 실행마다 회귀를 잡는다.

기준선을 1.0이 아니라 0.9로 둔 이유: 데이터셋에 더 어려운 케이스를 추가하는 것이
빌드를 즉시 깨뜨리는 일이 되면, 데이터셋을 쉬운 케이스로만 유지하려는 압력이 생긴다.
평가셋은 늘어나야 하는 자산이므로 약간의 여유를 둔다.
"""
import unittest

from evals.run_routing_eval import DATASET, evaluate, load_dataset


MIN_EXACT_MATCH = 0.9


class RoutingEvalGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.summary = evaluate(load_dataset(DATASET))

    def test_dataset_is_not_empty(self) -> None:
        self.assertGreaterEqual(self.summary["cases"], 20)

    def test_exact_match_above_baseline(self) -> None:
        failures = [r for r in self.summary["results"] if not r["exact"]]
        detail = "\n".join(
            f"  {r['id']}: 기대 {r['expected']} / 실제 {r['actual']}" for r in failures
        )
        self.assertGreaterEqual(
            self.summary["exact_match"],
            MIN_EXACT_MATCH,
            f"라우팅 정확도 회귀:\n{detail}",
        )

    def test_every_domain_is_covered_by_the_dataset(self) -> None:
        """평가셋이 특정 도메인을 빠뜨리면 그 도메인의 회귀를 못 잡는다."""
        for domain, metric in self.summary["per_domain"].items():
            with self.subTest(domain=domain):
                self.assertGreater(metric["support"], 0)


if __name__ == "__main__":
    unittest.main()
