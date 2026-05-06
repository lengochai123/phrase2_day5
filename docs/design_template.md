# Design Template

## Problem

Xây dựng một **research assistant tự động** có khả năng nhận một câu hỏi phức tạp, tìm kiếm thông tin từ nhiều nguồn, tổng hợp và phân tích, rồi trả về một bài viết có cấu trúc kèm trích dẫn. Hệ thống cần so sánh hiệu quả giữa single-agent và multi-agent workflow.

## Why multi-agent?

Single-agent gộp toàn bộ logic vào một LLM call duy nhất:
- Không thể kiểm soát chất lượng từng bước (search → analyze → write).
- Không có tracing rõ ràng để debug khi output kém.
- Không thể tái sử dụng hoặc thay thế từng phần (ví dụ: đổi search provider).
- Quality score của single-agent đạt 4.9/10 so với 8.3/10 của multi-agent (benchmark thực tế).

Multi-agent cho phép mỗi bước có prompt riêng, kiểm soát riêng, và có thể parallel hóa trong tương lai.

## Agent roles

| Agent | Responsibility | Input | Output | Failure mode |
|---|---|---|---|---|
| Supervisor | Điều phối, quyết định agent nào chạy tiếp | `ResearchState` hiện tại | route: `researcher / analyst / writer / done` | Vòng lặp vô hạn → fix: `max_iterations` hard stop |
| Researcher | Tìm kiếm và tóm tắt nguồn thông tin | `state.request.query` | `state.research_notes`, `state.sources` | Search API timeout → fix: mock fallback + retry |
| Analyst | Phân tích và rút ra insight từ research | `state.research_notes` | `state.analysis_notes` | Notes trống → fix: validate trước khi gọi LLM |
| Writer | Viết bài hoàn chỉnh kèm trích dẫn | `state.research_notes` + `state.analysis_notes` | `state.final_answer` | Thiếu analysis → fix: writer kiểm tra và raise nếu thiếu |
| Critic (optional) | Đánh giá chất lượng output, cho điểm 0-10 | `state.final_answer` | quality score trong metadata | Score parse fail → fix: regex với fallback = None |

## Shared state

`ResearchState` (Pydantic model) chứa:

| Field | Type | Lý do cần |
|---|---|---|
| `request` | `ResearchQuery` | Lưu query gốc để truyền xuyên suốt pipeline |
| `sources` | `list[SearchResult]` | Researcher ghi, Writer đọc để tạo trích dẫn |
| `research_notes` | `str` | Handoff từ Researcher → Analyst → Writer |
| `analysis_notes` | `str` | Handoff từ Analyst → Writer |
| `final_answer` | `str` | Output cuối, Supervisor dùng để detect `done` |
| `route_history` | `list[str]` | Debug: theo dõi toàn bộ luồng routing |
| `agent_results` | `list[AgentResult]` | Lưu token, cost, content của từng agent |
| `errors` | `list[str]` | Ghi lỗi không fatal để report sau |
| `iteration` | `int` | Supervisor dùng để enforce `max_iterations` |
| `trace_events` | `list[dict]` | Observability: timeline sự kiện |

## Routing policy

```
START
  └─► Supervisor
        ├─ research_notes empty?          ──► Researcher ──┐
        ├─ research ✓, analysis empty?    ──► Analyst    ──┤
        ├─ research ✓, analysis ✓,                         │
        │  final_answer empty?            ──► Writer     ──┤
        └─ final_answer ✓                ──► DONE           │
                                                            │
              ◄─────────────────── loop back ───────────────┘
              (max 10 iterations trước khi force DONE)
```

Cài đặt: rule-based thuần túy trong `SupervisorAgent._decide()`, không dùng LLM để tránh non-determinism trong mock mode.

## Guardrails

- **Max iterations:** 10 (config `MAX_ITERATIONS` trong `.env`), Supervisor force `done` và ghi error khi vượt ngưỡng.
- **Timeout:** Mỗi LLM call dùng `tenacity` retry 3 lần với exponential backoff (1s → 60s cap).
- **Retry:** `@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=60))` trên `LLMClient.complete()`.
- **Fallback:** Nếu OpenAI API không có key → mock LLM tự động; nếu LangGraph unavailable → state-machine Python loop.
- **Validation:** Analyst và Writer kiểm tra input notes không rỗng trước khi gọi LLM; CriticAgent parse score với regex có fallback.

## Benchmark plan

**Queries thử nghiệm:**
- `"Research GraphRAG state-of-the-art and write a 500-word summary"`
- `"Compare LangChain vs LlamaIndex for production RAG systems"`
- `"What are the key failure modes in multi-agent LLM systems?"`

**Metrics:**

| Metric | Cách đo | Kết quả thực tế |
|---|---|---|
| Latency | `perf_counter()` wall-clock | single: 0.05s / multi: 0.54s |
| Cost | token usage × price per 1K | single: $0 (mock) / multi: $0.0004 |
| Quality | heuristic 0-10 (final_answer + notes + citations) | single: 4.9 / multi: 8.3 |
| Citation coverage | số source title xuất hiện trong answer | embedded trong quality score |
| Failure rate | `len(state.errors) / total_queries` | 0 errors trong benchmark |

**Expected outcome:** Multi-agent score cao hơn ≥ 3 điểm so với single-agent do có research_notes (+2) và analysis_notes (+2) riêng biệt.
