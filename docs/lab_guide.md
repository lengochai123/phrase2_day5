# Lab Guide: Multi-Agent Research System

## Scenario

Bạn cần xây dựng một research assistant có thể nhận câu hỏi dài, tìm thông tin, phân tích và viết câu trả lời cuối cùng. Lab yêu cầu so sánh hai cách làm:

1. **Single-agent baseline**: một agent làm toàn bộ.
2. **Multi-agent workflow**: Supervisor điều phối Researcher, Analyst, Writer.

## Quy tắc quan trọng

- Không thêm agent nếu không có lý do rõ ràng.
- Mỗi agent phải có responsibility riêng.
- Shared state phải đủ rõ để debug.
- Phải có trace hoặc log cho từng bước.
- Phải benchmark, không chỉ nhìn output bằng cảm tính.

## Milestone 1: Baseline

File gợi ý:

- `src/multi_agent_research_lab/cli.py`
- `src/multi_agent_research_lab/services/llm_client.py`

**Đã implement:** `LLMClient.complete()` gọi OpenAI API thật với retry 3 lần
(exponential backoff). Nếu không có API key, tự động dùng mock LLM sinh
response dựa trên keyword của prompt. `SearchClient` tương tự: dùng Tavily
nếu có key, fallback về mock knowledge base gồm 8 document với keyword scoring.

## Milestone 2: Supervisor

File gợi ý:

- `src/multi_agent_research_lab/agents/supervisor.py`
- `src/multi_agent_research_lab/graph/workflow.py`

**Đã implement:** Routing policy rule-based thuần túy trong `_decide()`:

| Trạng thái state | Route tiếp theo |
|---|---|
| `research_notes` trống | `researcher` |
| Có research, chưa có analysis | `analyst` |
| Có cả hai, chưa có final_answer | `writer` |
| Có `final_answer` | `done` |
| Vượt `max_iterations` (10) | `done` + ghi error |

**Quyết định thiết kế:**
- Khi nào gọi Researcher? → `research_notes` còn trống.
- Khi nào gọi Analyst? → Có research nhưng chưa có analysis.
- Khi nào gọi Writer? → Có đủ cả research lẫn analysis.
- Khi nào stop? → `final_answer` đã tồn tại.
- Nếu agent fail? → Ghi vào `state.errors`, tiếp tục; nếu vượt max_iterations thì force done.

## Milestone 3: Worker agents

File gợi ý:

- `agents/researcher.py`
- `agents/analyst.py`
- `agents/writer.py`

**Đã implement:**

- **Researcher**: Gọi `SearchClient.search()` → ghi `state.sources`, sau đó gọi LLM
  để tổng hợp thành `state.research_notes`.
- **Analyst**: Đọc `state.research_notes` → gọi LLM → ghi `state.analysis_notes`
  (insight, pattern, implication).
- **Writer**: Đọc cả research + analysis → gọi LLM → ghi `state.final_answer`
  kèm References section.

Mỗi agent ghi `AgentResult` vào `state.agent_results` với token count và cost_usd.

## Milestone 4: Trace và benchmark

File gợi ý:

- `observability/tracing.py`
- `evaluation/benchmark.py`
- `evaluation/report.py`

**Đã implement:**

- `trace_span()` là context manager ghi start_time, duration, status, error cho
  mỗi bước vào global span list.
- `reset_run()` tạo run_id mới và xóa spans cũ trước mỗi benchmark run.
- `export_trace_json()` xuất toàn bộ spans ra file JSON.

Kết quả benchmark thực tế:

| Metric | Single-agent | Multi-agent |
|---|---:|---:|
| Latency (s) | 0.05 | 0.54 |
| Cost (USD) | — | $0.0004 |
| Quality (0-10) | 4.9 | 8.3 |
| Sources | 5 | 5 |
| Errors | 0 | 0 |

## Exit ticket

**Câu 1: Case nào nên dùng multi-agent? Vì sao?**

Nên dùng multi-agent khi task phức tạp, cần nhiều bước xử lý chuyên biệt:
- **Research pipeline**: Search → Analyze → Write cần từng bước chuyên sâu,
  một LLM call không đủ context window hoặc độ chính xác.
- **Quality control**: Cần CriticAgent review output trước khi trả về user.
- **Parallel workload**: Nhiều sub-task có thể chạy song song (nhiều Researcher
  theo từng domain khác nhau).
- **Traceability quan trọng**: Cần biết chính xác bước nào tốn bao nhiêu, sai
  ở đâu để debug và cải thiện.

Benchmark cho thấy multi-agent đạt quality 8.3/10 so với 4.9/10 của
single-agent — chênh lệch 3.4 điểm nhờ có research_notes và analysis_notes
riêng biệt.

**Câu 2: Case nào không nên dùng multi-agent? Vì sao?**

Không nên dùng multi-agent khi:
- **Task đơn giản**: Câu hỏi factual ngắn (ví dụ "thủ đô của Pháp là gì?")
  không cần pipeline phức tạp — overhead routing làm latency tăng 10× (0.05s
  vs 0.54s) mà chất lượng không cải thiện đáng kể.
- **Latency là ưu tiên số 1**: Real-time chat cần response < 1s; mỗi agent
  hop thêm ~0.1–0.2s.
- **Budget thấp**: Mỗi agent gọi LLM riêng → cost nhân lên theo số agent.
- **Team nhỏ, ít kinh nghiệm**: Debug multi-agent phức tạp hơn nhiều; nếu
  team chưa có tracing/observability tốt thì lỗi sẽ rất khó tìm.
- **Task không phân tách được**: Nếu không thể tách rõ trách nhiệm của từng
  agent (overlap quá nhiều), thêm agent chỉ tăng complexity mà không tăng
  chất lượng.
