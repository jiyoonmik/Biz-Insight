"""라우팅 평가 러너 (Tier 1).

Supervisor를 LLM이 아니라 규칙으로 만든 판단(README §2.1)의 직접적인 대가로,
라우팅 정확도를 LLM 호출도 API 키도 없이 매 커밋 측정할 수 있다. 실행 시간은
초 단위이고 결과는 완전히 재현된다. 그래서 이 평가는 CI 게이트로 쓴다.

측정 대상은 '요청 문장 → 선택된 분석 도메인 집합'이다. 도메인 선택이 틀리면
그 하류의 도구 선택, 수집 근거, 리포트 구성이 전부 함께 틀어지므로 파이프라인에서
가장 앞이자 가장 싸게 잴 수 있는 품질 지점이다.

사용법:
    python -m evals.run_routing_eval
    python -m evals.run_routing_eval --min-exact-match 0.8   # CI 게이트
    python -m evals.run_routing_eval --verbose               # 실패 케이스 상세
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.agents.supervisor import _build_plan


DATASET = Path(__file__).parent / "datasets" / "routing.jsonl"
ALL_DOMAINS = ["overview", "financial", "credit", "investment", "review", "risk"]


def load_dataset(path: Path) -> list[dict]:
    cases = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line and not line.startswith("//"):
                cases.append(json.loads(line))
    return cases


def evaluate(cases: list[dict]) -> dict:
    results = []
    for case in cases:
        domains, tools, needs_analysis = _build_plan(case["query_type"], case["request"])
        expected = set(case["expected_domains"])
        actual = set(domains)
        results.append(
            {
                "id": case["id"],
                "request": case["request"],
                "query_type": case["query_type"],
                "expected": sorted(expected),
                "actual": sorted(actual),
                "exact": expected == actual,
                "jaccard": len(expected & actual) / len(expected | actual) if expected | actual else 1.0,
                "missing": sorted(expected - actual),
                "extra": sorted(actual - expected),
                "tool_count": len(tools),
                "needs_analysis": needs_analysis,
            }
        )

    per_domain = {}
    for domain in ALL_DOMAINS:
        tp = sum(1 for r in results if domain in r["expected"] and domain in r["actual"])
        fp = sum(1 for r in results if domain not in r["expected"] and domain in r["actual"])
        fn = sum(1 for r in results if domain in r["expected"] and domain not in r["actual"])
        precision = tp / (tp + fp) if tp + fp else 1.0
        recall = tp / (tp + fn) if tp + fn else 1.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_domain[domain] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": tp + fn,
        }

    total = len(results)
    return {
        "cases": total,
        "exact_match": round(sum(1 for r in results if r["exact"]) / total, 4) if total else 0.0,
        "mean_jaccard": round(sum(r["jaccard"] for r in results) / total, 4) if total else 0.0,
        "per_domain": per_domain,
        "results": results,
    }


def render(summary: dict, verbose: bool) -> str:
    lines = [
        "# Routing Eval",
        "",
        f"- 케이스: {summary['cases']}건",
        f"- 완전 일치(exact match): **{summary['exact_match']:.1%}**",
        f"- 평균 자카드 유사도: {summary['mean_jaccard']:.3f}",
        "",
        "| domain | precision | recall | f1 | support |",
        "|---|---|---|---|---|",
    ]
    for domain, metric in summary["per_domain"].items():
        lines.append(
            f"| {domain} | {metric['precision']:.2f} | {metric['recall']:.2f} "
            f"| {metric['f1']:.2f} | {metric['support']} |"
        )

    failures = [r for r in summary["results"] if not r["exact"]]
    if failures:
        lines += ["", f"## 불일치 {len(failures)}건", ""]
        for row in failures if verbose else failures[:10]:
            lines.append(f"- `{row['id']}` ({row['query_type']}) {row['request']}")
            lines.append(f"  - 기대: {row['expected']} / 실제: {row['actual']}")
            if row["missing"]:
                lines.append(f"  - 누락: {row['missing']}")
            if row["extra"]:
                lines.append(f"  - 과선택: {row['extra']}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Supervisor 라우팅 평가 (LLM 미사용)")
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--min-exact-match", type=float, default=None,
                        help="이 값 미만이면 exit code 1 (CI 게이트)")
    parser.add_argument("--json", action="store_true", help="요약을 JSON으로 출력")
    parser.add_argument("--verbose", action="store_true", help="불일치 전체 출력")
    args = parser.parse_args()

    cases = load_dataset(args.dataset)
    summary = evaluate(cases)

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(render(summary, args.verbose))

    if args.min_exact_match is not None and summary["exact_match"] < args.min_exact_match:
        print(
            f"\n❌ exact match {summary['exact_match']:.1%} < 기준 {args.min_exact_match:.1%}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
