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
