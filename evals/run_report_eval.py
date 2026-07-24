"""리포트 평가 러너 (Tier 2).

Tier 1(라우팅)과 달리 이쪽은 실제로 그래프를 돌리므로 API 키와 시간이 든다.
그래서 CI 게이트가 아니라 변경 전후를 비교하는 수동 회귀 측정용이다.

**채점자로 LLM을 쓰지 않는다.** LLM-as-judge를 붙이면 채점자 자신이 흔들려서
"프롬프트를 고쳤더니 점수가 올랐다"와 "채점자가 그날 후하게 굴었다"를 구분할 수
없게 된다. 회귀를 재는 것이 목적이므로, 재현 가능한 것만 잰다.

- 근거 대조율: Verifier가 이미 계산한 값 (도구 반환값 대비 본문 수치 일치 비율)
- 구조 점검: 리포트가 반드시 포함해야 하는 요소가 실제로 있는지 정규식 확인
- 실행 지표: 에이전트 경로, 도구 호출 수, 재검토 발생 여부, 오류, 소요 시간

사용법:
    python -m evals.run_report_eval --dry-run     # LLM 없이 데이터셋/라우팅만 점검
    python -m evals.run_report_eval --limit 2     # 실제 실행 (GOOGLE_API_KEY 필요)
"""
from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any


DATASET = Path(__file__).parent / "datasets" / "reports.jsonl"
RESULTS_DIR = Path(__file__).parent / "results"

# 리포트가 갖춰야 할 구조 요건. Synthesis 프롬프트의 작성 지침과 1:1로 대응한다.
# 지침을 넣어 두고 지켜졌는지 재지 않으면 프롬프트는 조용히 무너진다.
RUBRIC = {
    "stock_code_명시": re.compile(r"\b\d{6}\b"),
    "기준연도_명시": re.compile(r"(19|20)\d{2}\s*년|회계연도|기준\s*연도"),
    "SWOT_포함": re.compile(r"SWOT", re.IGNORECASE),
    "리스크_섹션": re.compile(r"리스크|위험"),
    "신용_또는_재무_근거": re.compile(r"신용등급|부채비율|영업이익|매출|유동비율"),
    "근거대조_섹션": re.compile(r"근거 대조"),
}


def load_dataset(path: Path) -> list[dict]:
    cases = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line and not line.startswith("//"):
                cases.append(json.loads(line))
    return cases


def score_report(report: str, trace: dict[str, Any]) -> dict[str, Any]:
    rubric = {name: bool(pattern.search(report or "")) for name, pattern in RUBRIC.items()}
    grounding = trace.get("grounding") or {}
    usage = trace.get("usage") or {}
    return {
        "rubric": rubric,
        "llm_calls": usage.get("llm_calls", 0),
        "total_tokens": usage.get("total_tokens", 0),
        "llm_seconds": usage.get("llm_seconds", 0),
        "throttle_seconds": usage.get("throttle_seconds", 0),
        "calls_by_agent": usage.get("calls_by_agent", {}),
        "rubric_pass_rate": round(sum(rubric.values()) / len(rubric), 4),
        "grounding_coverage": grounding.get("coverage"),
        "numbers_checked": grounding.get("checked", 0),
        "numbers_matched": grounding.get("matched", 0),
        "evidence_count": trace.get("evidence_count", 0),
        "canonical_facts": grounding.get("canonical_facts", 0),
        "dynamic_facts": grounding.get("dynamic_facts", 0),
        "agents": trace.get("agents", []),
        "tool_calls": len(trace.get("tool_calls", []) or []),
        "domains": trace.get("requested_domains", []),
        "revised": any("reviewer" in str(a) for a in trace.get("agents", [])),
        "errors": trace.get("errors", []) or [],
        "report_chars": len(report or ""),
    }


def run_case(case: dict) -> dict[str, Any]:
    from src.graph import generate_ai_report_result

    started = time.time()
    result = generate_ai_report_result(
        case["company"], case.get("query_type", "full_report"), case.get("request")
    )
    elapsed = time.time() - started
    scored = score_report(result["report"], result["trace"])
    scored.update({"id": case["id"], "company": case["company"], "elapsed_sec": round(elapsed, 1)})
    return scored


def dry_run(cases: list[dict]) -> list[dict[str, Any]]:
    """LLM 없이 데이터셋 정합성과 라우팅 계획만 점검한다."""
    from src.agents.supervisor import _build_plan

    rows = []
    for case in cases:
        domains, tools, needs_analysis = _build_plan(
            case.get("query_type", "full_report"), case.get("request") or ""
        )
        rows.append(
            {
                "id": case["id"],
                "company": case["company"],
                "domains": domains,
                "tool_count": len(tools),
                "needs_analysis": needs_analysis,
            }
        )
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [r for r in rows if r.get("grounding_coverage") is not None]
    coverages = [r["grounding_coverage"] for r in scored]
    return {
        "cases": len(rows),
        "mean_grounding_coverage": round(sum(coverages) / len(coverages), 4) if coverages else None,
        "mean_rubric_pass_rate": round(
            sum(r.get("rubric_pass_rate", 0) for r in rows) / len(rows), 4
        ) if rows else None,
        "mean_evidence_count": round(
            sum(r.get("evidence_count", 0) for r in rows) / len(rows), 1
        ) if rows else None,
        "cases_with_errors": sum(1 for r in rows if r.get("errors")),
        "mean_elapsed_sec": round(
            sum(r.get("elapsed_sec", 0) for r in rows) / len(rows), 1
        ) if rows else None,
        "mean_llm_calls": round(
            sum(r.get("llm_calls", 0) for r in rows) / len(rows), 1
        ) if rows else None,
        "mean_total_tokens": round(
            sum(r.get("total_tokens", 0) for r in rows) / len(rows)
        ) if rows else None,
        "mean_throttle_sec": round(
            sum(r.get("throttle_seconds", 0) for r in rows) / len(rows), 1
        ) if rows else None,
    }


def render(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Report Eval",
        "",
        f"- 케이스: {summary['cases']}건",
        f"- 평균 근거 대조율: {summary['mean_grounding_coverage']}",
        f"- 평균 구조 요건 충족률: {summary['mean_rubric_pass_rate']}",
        f"- 평균 근거 원장 수집량: {summary['mean_evidence_count']}건",
        f"- 오류가 기록된 케이스: {summary['cases_with_errors']}건",
        f"- 평균 소요: {summary['mean_elapsed_sec']}초 "
        f"(rate limit 대기 {summary['mean_throttle_sec']}초 포함)",
        f"- 평균 LLM 호출: {summary['mean_llm_calls']}회 / 토큰 {summary['mean_total_tokens']:,}",
        "",
        "| id | 도메인 | 도구 | 원장 | 대조율 | 구조 | LLM | 토큰 | 오류 | 소요(s) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        coverage = row.get("grounding_coverage")
        lines.append(
            f"| {row['id']} | {','.join(row.get('domains', []))} | {row.get('tool_calls', 0)} "
            f"| {row.get('evidence_count', 0)} "
            f"| {'n/a' if coverage is None else f'{coverage:.0%}'} "
            f"| {row.get('rubric_pass_rate', 0):.0%} | {row.get('llm_calls', 0)} "
            f"| {row.get('total_tokens', 0):,} | {len(row.get('errors', []))} "
            f"| {row.get('elapsed_sec', 0)} |"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="리포트 품질 회귀 평가 (LLM 채점자 없음)")
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="LLM 없이 라우팅 계획만 점검")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    cases = load_dataset(args.dataset)
    if args.limit:
        cases = cases[: args.limit]

    if args.dry_run:
        rows = dry_run(cases)
        print(json.dumps({"dry_run": True, "cases": rows}, ensure_ascii=False, indent=2))
        return 0

    rows = []
    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] {case['id']} · {case['company']} 실행 중...", flush=True)
        try:
            rows.append(run_case(case))
        except Exception as exc:  # 한 건 실패가 전체 평가를 날리지 않게 한다
            rows.append({"id": case["id"], "company": case["company"], "errors": [f"run_failed: {exc}"]})

    summary = summarize(rows)
    print()
    print(render(summary, rows))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = args.out or RESULTS_DIR / f"report_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(
        json.dumps({"summary": summary, "results": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n결과 저장: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
