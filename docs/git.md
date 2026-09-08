# Hướng dẫn làm việc trên fork chung

Repo gốc: `khacnambn/IsaacSim_TransformerBipedal`
Fork chung của team: `cat2I/TransformBipedal_Simulation`

## SETUP (làm 1 lần duy nhất)

**Việc của cat2I — thêm dev vào fork:**
```bash
gh api repos/cat2I/TransformBipedal_Simulation/collaborators/<username-dev> -X PUT
```
Cấp quyền push trực tiếp vào fork chung cho từng dev.

**Việc của mỗi dev (kể cả khi setup máy mới):**
```bash
git clone https://github.com/cat2I/TransformBipedal_Simulation.git
cd TransformBipedal_Simulation
git remote add upstream https://github.com/khacnambn/IsaacSim_TransformerBipedal.git
```
`git clone` tải fork chung của team về máy — đây là nơi cả nhóm cùng làm việc.
`git remote add upstream` thêm đường trỏ tới repo gốc của anh Nam, dùng để lấy cập nhật khi cần.

## WORKFLOW HẰNG NGÀY

**1. Trước khi code — lấy code mới nhất của team:**
```bash
git checkout main
git pull origin main
```

**2. Tạo nhánh riêng cho việc đang làm** (không code thẳng trên `main`):
```bash
git checkout -b ten-tinh-nang
```

**3. Code xong, lưu lại:**
```bash
git add -A
git commit -m "mô tả ngắn gọn đã sửa gì"
```

**4. Đẩy nhánh lên fork chung để người khác thấy:**
```bash
git push -u origin ten-tinh-nang
```

**5. Mở PR nội bộ trong chính fork** (branch của bạn → `main` của fork) trên GitHub, để 2 người còn lại review trước khi gộp. Không tự merge thẳng vào `main` một mình.

**6. Khi cả nhóm thống nhất xong trong `main` của fork** — người đại diện (cat2I) mở PR thật gửi cho anh Nam:
```bash
gh pr create --repo khacnambn/IsaacSim_TransformerBipedal --base main --head cat2I:main
```

**7. Khi anh Nam cập nhật repo gốc trong lúc nhóm đang làm** (chạy định kỳ để không bị lệch quá xa):
```bash
git fetch upstream
git merge upstream/main
git push origin main
```

## ĐỌC / KIỂM TRA XUNG ĐỘT / GỘP NHÁNH CỦA ĐỒNG ĐỘI

Dùng khi đồng đội (VD: Vinh) đã push 1 nhánh riêng lên fork chung, và bạn muốn xem code đó, kiểm tra có đụng code mình không, rồi mới gộp vào `main`.

**1. Cập nhật danh sách nhánh mới nhất từ fork chung:**
```bash
git fetch origin
```
`fetch` chỉ tải thông tin về, không đổi gì trên máy bạn — chạy lúc nào cũng an toàn. Nếu Source Control (VSCode) chưa thấy nhánh mới của đồng đội, 99% là do chưa fetch — nhánh đã có trên GitHub rồi nhưng máy bạn chưa biết.

**2. Trước khi chuyển nhánh — đảm bảo việc đang làm dở đã được lưu:**
```bash
git status
```
Nếu thấy có thay đổi chưa commit, lưu tạm lại trước (chọn 1 trong 2):
```bash
git add -A && git commit -m "wip: mô tả ngắn"   # lưu hẳn
git stash -u                                      # hoặc cất tạm, lấy lại sau bằng git stash pop
```
Bỏ qua bước này rồi đổi nhánh có thể làm thay đổi bị mang lộn xộn sang nhánh khác, hoặc git chặn không cho đổi.

**3. Qua nhánh của đồng đội để đọc / chạy thử** (chưa gộp gì cả):
```bash
git checkout -b ten-nhanh origin/ten-nhanh
```
Tạo 1 nhánh local nội dung y hệt `origin/ten-nhanh`, để bạn `cd` vào đọc code, chạy thử thoải mái. Đọc xong, quay lại nhánh chính:
```bash
git checkout main
```

**4. Kiểm tra trước xem gộp vào có bị xung đột không** (chỉ thử, không đổi gì, an toàn 100%):
```bash
git merge-tree --write-tree HEAD origin/ten-nhanh
```
- Kết quả chỉ ra 1 mã hash duy nhất → sạch, gộp được luôn, không lo gì.
- Có dòng `CONFLICT (content): Merge conflict in <tên file>` → 2 bên cùng sửa 1 chỗ theo cách khác nhau, cần bàn với đồng đội trước khi gộp thật.

**5. Gộp nhánh đồng đội vào `main` của bạn:**
```bash
git checkout main
git merge origin/ten-nhanh
git push origin main
```
Sau khi push xong, báo đồng đội `git fetch origin` rồi `git pull` để lấy phần bạn vừa gộp/thêm.

**6. Nếu bước 4 báo có xung đột thật, lúc merge ở bước 5 sẽ dừng lại** — xử lý như sau:
- VSCode hiện các file lỗi trong Source Control, dưới mục **"Merge Changes"**
- Mở từng file, chọn **Accept Current / Accept Incoming / Accept Both** trong merge editor 3 cột
- Xong hết các file thì lưu lại:
```bash
git add -A
git commit
```
- Muốn hủy gộp, quay về như trước khi merge (chưa mất gì cả):
```bash
git merge --abort
```

**Lưu ý:** nếu 1 nhánh tồn tại ở cả `origin` (fork chung) lẫn `upstream` (repo anh Nam) với cùng tên, luôn ghi rõ `origin/ten-nhanh` hay `upstream/ten-nhanh` khi gõ lệnh — 2 remote khác nhau, nội dung có thể lệch nhau dù trùng tên.
