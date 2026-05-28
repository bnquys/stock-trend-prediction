# Quy tắc làm việc với dự án (Project Rules)

## 1. Trước khi thực hiện thay đổi lớn (Big Refactor / Tính năng mới phức tạp)
- Bạn PHẢI tự động chạy lệnh Terminal sau để cập nhật bản đồ dự án:
  `uvx repomix --include "src/**/*"`
- Đọc file `repomix-output.txt` (hoặc định dạng .xml tương ứng) vừa được tạo ra để lấy ngữ cảnh toàn diện nhất.
- Sau khi đọc xong, hãy lập kế hoạch sửa đổi và thảo luận với người dùng trước khi code.

## 2. Quản lý Token và File
- Tuyệt đối không dùng lệnh `cat` hoặc đọc thủ công từng file trong thư mục `src` nếu không cần thiết. Hãy tận dụng file output của repomix để tiết kiệm token.
- Sau khi hoàn thành toàn bộ nhiệm vụ và refactor xong, hãy chạy lại lệnh `uvx repomix` một lần nữa để cập nhật trạng thái mới nhất cho hệ thống.