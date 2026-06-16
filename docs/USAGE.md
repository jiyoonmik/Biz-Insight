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

로컬 RAG 벡터스토어 생성:

```bash
UV_PROJECT_ENVIRONMENT=bizinsight uv run --no-sync python -m src.rag.build_vectorstore build \
  --data-dir data \
  --out-dir data/vectorstore \
  --max-rows-per-doc 80 \
  --max-features 100000
```

RAG 검색 예시:

```bash
UV_PROJECT_ENVIRONMENT=bizinsight uv run --no-sync python -m src.rag.build_vectorstore query \
  "신용등급 재무 안정성 부채 상환능력" \
  --entity 삼성전자 \
  --top-k 3
```

## Development

Install dependencies with `uv` using the project environment used by this repository.

```bash
UV_PROJECT_ENVIRONMENT=bizinsight uv sync
```

Run tests:

```bash
UV_PROJECT_ENVIRONMENT=bizinsight uv run --no-sync python -m unittest discover -s tests
```

When changing the data model, update the flow in this order:

1. canonical builder
2. context query function
3. LangChain tool or API endpoint
4. Streamlit or agent prompt surface
