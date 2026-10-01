# AGENTS.md

## 0. Quy trình làm việc (Multi-Agent Workflow Protocol)
Đây là quy tắc bắt buộc dành cho các AI Agents (Claude & Gemini) khi tương tác với dự án này:
- **Vai trò của Claude**: Đóng vai trò Tech Lead, Planner và QA. Nhiệm vụ của bạn là phân tích yêu cầu, lập kế hoạch vào file `PLAN.md`, và **review code**. Claude KHÔNG trực tiếp viết/sửa code.
- **Vai trò của Gemini (Antigravity CLI)**: Đóng vai trò Coder (Thợ code). Gemini sẽ đọc `PLAN.md` và trực tiếp thực thi việc sửa file, dặc biệt lưu ý agy không được sửa `PLAN.md`.
- **Cách thức Review (Dành cho Claude)**: Việc nghiệm thu (review) CHỈ dựa trên nội dung `git diff` do User cung cấp để đối chiếu với các Acceptance Criteria trong `PLAN.md`. Tuyệt đối không yêu cầu User dán lại toàn bộ source code. Nếu có gì agy làm không đúng thì giao cho agy làm lại trong 'FIX_AFTER_DIFF.md'. Đối với cả `PLAN.md` và 'FIX_AFTER_DIFF.md' mục nào agy làm tốt rồi thì tích pass, chưa tốt thì là chưa pass; PLAN thì để luồng công việc chính còn FIX_AFTER_DIFF thì là để sửa những gì agy làm sai, viết theo lần lượt và không xóa cái nào để tới cuối có thứ tổng hợp những gì agy làm sai.
- **Nguyên tắc "Thợ code" (Dành cho Gemini)**: Cấm không được tự ý refactor hay thay đổi bất kỳ logic/file nào nằm ngoài phạm vi đã vạch ra trong `PLAN.md`.

## 1. Context chung
Đây là thư mục gốc chứa toàn bộ mã nguồn cho dự án Reinforcement Learning (RL) Pendulum.
Dự án bao gồm 2 phần chính:
- **Thư mục Nhúng (Embedded)**: Code chạy trực tiếp trên vi điều khiển/thiết bị thực nhưng nằm ở folder khác.
- **Thư mục Mô phỏng (Simulation)**: Ngay tịa đây là code chạy mô phỏng môi trường RL và huấn luyện mô hình, xuất trọng số ONNX để nhúng vào vi điều khiển.

## 2. Quy tắc Coding chung
- **Test trung thực**: Không tự ý sửa điều kiện test, không hardcode để "vượt rào" test. Nếu có logic chưa chắc chắn, hãy để lại comment `TODO: [Mô tả vấn đề]`.
- **Tính đồng bộ**: Luôn lưu ý sự tương thích giữa Simulation và Embedded (đặc biệt là State Space, Action Space, Reward struct, và Communication Protocol).
- **Coding Style**: Tôn trọng convention đang có sẵn trong từng thư mục. Viết docstring rõ ràng cho các hàm liên quan đến thuật toán RL hoặc phần cứng.
