# Benchmark Report

| Run | Latency (s) | Cost (USD) | Quality | Notes |
|---|---:|---:|---:|---|
| single-agent | 0.05 |  | 4.9 | routes= | sources=5 | errors=0 |
| multi-agent | 0.54 | 0.0004 | 8.3 | routes=researcher -> analyst -> writer -> done | sources=5 | errors=0 |

---

## Failure Modes & Fixes

Trong quá trình xây dựng hệ thống, năm lỗi chính đã được phát hiện và xử lý:

### 1. Vòng lặp vô hạn (Infinite Loop)

**Nguyên nhân:** Nếu Supervisor tiếp tục định tuyến về các agent mà không bao giờ đạt đến trạng thái `done`, workflow sẽ chạy mãi mãi.

**Fix:** Thêm biến `max_iterations` trong `SupervisorAgent.run()`. Khi vượt ngưỡng, supervisor buộc route về `done` và ghi lỗi vào `state.errors`:
```python
if state.iteration >= self._settings.max_iterations:
    next_route = "done"
    state.errors.append("Stopped at max_iterations=...")
```

---

### 2. Mock LLM định tuyến sai (Wrong Routing in Mock Mode)

**Nguyên nhân:** Khi dùng LLM để quyết định route, mock LLM nhận prompt chứa từ khóa như `"research_notes"` và `"analysis_notes"` rồi trả về `"writer"` ngay cả khi state còn trống.

**Fix:** Loại bỏ hoàn toàn LLM call khỏi `SupervisorAgent._decide()`. Thay bằng logic rule-based thuần túy:
```python
def _decide(self, state):
    if state.final_answer:    return "done"
    if has_research and has_analysis: return "writer"
    if has_research:          return "analyst"
    return "researcher"
```

---

### 3. File bị cắt ngắn trên Windows/Linux mount (File Truncation)

**Nguyên nhân:** File Write tool trên Windows đồng bộ sang Linux mount bị mất nội dung (629 bytes thay vì ~3000 bytes). Ký tự Unicode như em-dash (`—`) cũng gây lỗi encoding.

**Fix:** Viết tất cả file Python qua bash heredoc để tránh qua lớp đồng bộ Windows↔Linux:
```bash
cat > file.py << 'PYEOF'
# nội dung file
PYEOF
```
Xác minh bằng `wc -c` và `python3 -m py_compile` sau mỗi lần ghi.

---

### 4. `StrEnum` không tồn tại trên Python 3.10

**Nguyên nhân:** `from enum import StrEnum` chỉ có từ Python 3.11 trở lên, gây `ImportError` trên Python 3.10.

**Fix:** Thêm conditional import với backport trong `schemas.py`:
```python
import sys
if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from enum import Enum
    class StrEnum(str, Enum):
        pass
```

---

### 5. LangGraph TypeError: Pydantic model vs plain dict

**Nguyên nhân:** LangGraph `StateGraph` yêu cầu các node nhận và trả về `dict`, nhưng code ban đầu truyền thẳng `ResearchState` (Pydantic model).

**Fix:** Thêm hai hàm wrapper `_dump()` / `_load()` trong mỗi node của LangGraph:
```python
def _dump(s: ResearchState) -> dict:
    return s.model_dump()

def _load(d: dict) -> ResearchState:
    return ResearchState.model_validate(d)

def node_researcher(d: dict) -> dict:
    return _dump(researcher.run(_load(d)))
```

---

## Kết luận

| Failure Mode | Tác động | Giải pháp |
|---|---|---|
| Infinite loop | Chạy mãi không dừng | `max_iterations` hard stop |
| Mock routing sai | Bỏ qua researcher/analyst | Rule-based `_decide()` |
| File truncation | Code bị mất | Bash heredoc + verify |
| StrEnum Python 3.10 | ImportError | Conditional backport |
| LangGraph TypeError | Workflow crash | `_dump`/`_load` wrappers |
