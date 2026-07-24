# Biz-Insight Usage Notes

이 문서는 로컬 실행, 환경변수, 데이터 산출물 생성처럼 프로젝트 관리자가 참고할 사용 방법을 정리합니다. 외부 소개용 내용은 루트 `README.md`를 기준으로 관리합니다.

## API

FastAPI 앱은 `src/api/main.py`에 있습니다.

주요 엔드포인트:

- `GET /health`
- `GET /companies/search?name=삼성전자`
- `GET /companies/{stock_code}`
- `GET /companies/{stock_code}/knowledge-context?year=2022`
- `POST /reports/company-analysis`

로컬 실행:

```bash
UV_PROJECT_ENVIRONMENT=bizinsight uv run --no-sync uvicorn src.api.main:app --reload
```

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

## Streamlit App

Streamlit 진입점은 루트의 `app.py`입니다. 기업명을 검색하고 분석 유형 또는 자연어 요청을 입력하면 LangGraph 멀티에이전트가 실행됩니다.

```bash
UV_PROJECT_ENVIRONMENT=bizinsight uv run --no-sync streamlit run app.py
```

포트폴리오 데모 배포는 전체 `data/` 대신 `data/demo` snapshot을 사용하는 것을 권장합니다.

```bash
BIZINSIGHT_DATA_DIR=data/demo \
BIZINSIGHT_APP_PASSWORD=your-demo-password \
UV_PROJECT_ENVIRONMENT=bizinsight uv run --no-sync streamlit run app.py
```

데모 배포 절차와 데이터 caveat은 `docs/DEMO_DEPLOYMENT.md`를 참고하세요. 데모 데이터는 2026-06-13 KST 기준의 작은 샘플이며, 삼성전자·현대자동차·카카오를 KOSPI 기준점으로, 케이아이엔엑스·씨앤씨인터내셔널·제이브이엠·덕산네오룩스·인텔리안테크를 KOSDAQ 성장기업 분석 대상으로 둡니다.

리포트 생성 후에는 화면의 `리포트 작성 과정 보기`에서 실행된 에이전트, 호출 도구, 데이터 출처 marker를 확인할 수 있습니다.

## Environment

`.env`에는 LLM 호출을 위한 `GOOGLE_API_KEY`가 필요합니다.

```text
GOOGLE_API_KEY=...
```

LangSmith tracing을 활성화하려면 `.env`에 아래 값을 추가합니다. `LANGSMITH_PROJECT`는 LangSmith에서 실행을 묶어 볼 프로젝트 이름입니다.

```text
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=...
LANGSMITH_PROJECT=biz-insight-local
# LANGSMITH_ENDPOINT=https://api.smith.langchain.com
```

환경변수가 설정되면 Streamlit의 `AI 리포트 생성 시작` 버튼에서 실행되는 LangGraph run이 LangSmith에 기록됩니다. 각 실행에는 `biz-insight-report` run name, `query_type:*` tag, 기업명과 사용자 요청 metadata가 포함됩니다.

## Build Data Artifacts

canonical 테이블 생성:

```bash
UV_PROJECT_ENVIRONMENT=bizinsight uv run --no-sync python -m src.canonical.build_all
```

dynamic 스토어(주가·거시·직원 후기) 생성. **후기 스토어는 세 소스(employee/b_/j_)를 리치 스키마로 통합**하므로, 후기 검색을 쓰려면 먼저 빌드해야 합니다:

```bash
UV_PROJECT_ENVIRONMENT=bizinsight uv run --no-sync python -m src.dynamic.refresh_dynamic_data
# 후기 스토어만: python -m src.dynamic.review_store
```

후기 semantic 검색은 `search_review_evidence(stock_code, query)`로 노출되며, 부분문자열이 아니라 문자 n-gram TF-IDF 유사도로 회사 후기를 정렬합니다(§2.12). scikit-learn이 없으면 부분문자열로 자동 폴백합니다.

## Development

Install dependencies with `uv` using the project environment used by this repository.

```bash
UV_PROJECT_ENVIRONMENT=bizinsight uv sync --extra dev
```

`data/canonical`은 파생 데이터라 저장소에 없습니다. 처음 클론했다면 테스트 전에 빌드하세요 (약 8초).

```bash
UV_PROJECT_ENVIRONMENT=bizinsight uv run --no-sync python -m src.canonical.build_all
```

Run tests (LLM 호출 없음, 약 20초):

```bash
UV_PROJECT_ENVIRONMENT=bizinsight uv run --no-sync python -m pytest tests -q
```

Run evals — 자세한 설계 배경은 `evals/README.md` 참고:

```bash
# Tier 1: 라우팅 평가 (LLM 불필요, CI 게이트와 동일)
UV_PROJECT_ENVIRONMENT=bizinsight uv run --no-sync python -m evals.run_routing_eval --verbose

# Tier 2: 리포트 평가 (GOOGLE_API_KEY 필요)
UV_PROJECT_ENVIRONMENT=bizinsight uv run --no-sync python -m evals.run_report_eval --dry-run
UV_PROJECT_ENVIRONMENT=bizinsight uv run --no-sync python -m evals.run_report_eval --limit 2
```

Rate limit 지연은 **LLM 호출 직후에만** 발생하며 환경변수로 조절합니다. 유료 쿼터나 로컬 반복 실행에서는 0으로 낮출 수 있습니다.

```text
BIZINSIGHT_POST_LLM_DELAY_SEC=1.5      # LLM 호출 후 대기 (기본 1.5, 0이면 비활성)
```

실행 비용(LLM 호출 수, 토큰, 모델 대기, rate limit 대기)은 Streamlit의 `실행 로그 및 데이터 출처 보기`와 Tier 2 평가 결과에 함께 기록됩니다.

When changing the data model, update the flow in this order:

1. canonical builder
2. context query function
3. LangChain tool or API endpoint
4. 근거 원장 추출기 (`src/agents/evidence.py`) — 새 도구가 새로운 행 모양을 반환하면 추가
5. Streamlit or agent prompt surface
