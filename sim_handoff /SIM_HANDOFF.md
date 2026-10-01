# Bàn giao cho bên sim: URDF xuất ngày 2026-09-30

## File đã xuất ở đâu

SW2URDF xuất lúc 18:26, vào thư mục nằm **cạnh** thư mục `Transformer` (không nằm bên trong):

```
D:\des\lab\Vinh&Minh\Transformer_VinhVersion\Transformer_VinhVersion\Transformer\OFFICIALdesign\
├─ urdf\OFFICIALdesign.urdf    ← file chính
├─ urdf\OFFICIALdesign.csv     (bảng của exporter, số mass sai, không dùng)
├─ meshes\*.STL                ← 13 mesh, mỗi link 1 file
├─ config\, launch\, package.xml, CMakeLists.txt   (dành cho ROS, Isaac không cần)
└─ export.log                  (không cần)
```

`OFFICIALdesign.SLDASM` được lưu lúc 18:50, sau khi xuất, nên cấu hình exporter và các hệ tọa độ mới đã được giữ lại.

Bản đóng gói sẵn cho Ubuntu: `D:\des\lab\Vinh&Minh\Transformer_VinhVersion\Transformer_VinhVersion\Transformer\sim_handoff.zip`. Trong đó có `OFFICIALdesign/urdf`, `OFFICIALdesign/meshes`, `inertial_report.csv/.md`, `MASS_INPUT_GUIDE.md` và file này.

## Cần đưa cho bên sim

1. `...\Transformer\OFFICIALdesign\urdf\OFFICIALdesign.urdf`
2. `...\Transformer\OFFICIALdesign\meshes\` (cả thư mục)
3. `...\Transformer\Transformer\tools\inertial_report.csv`: mass, trọng tâm và quán tính đúng cho 13 link, dùng để sửa URDF
4. `...\Transformer\Transformer\MASS_INPUT_GUIDE.md`: checklist cho bên sim và các giá trị cần ghi đè

## Vấn đề phát hiện được

### Phải sửa trước khi chạy sim

1. **Link `Bubright` có mass = 0 và inertia = 0.** Đây là lỗi của exporter; bản đối xứng `Bubleft` vẫn bình thường (0,0588). Link khối lượng 0 sẽ làm PhysX lỗi hoặc cho kết quả vô nghĩa. Giá trị đúng có trong `inertial_report.csv`.
2. **Tổng khối lượng trong URDF là 4,044 kg, đúng phải là 4,643 kg**, vì exporter bỏ qua override:

   | Link | Trong URDF (kg) | Đúng (kg) |
   |---|---|---|
   | Baselink | 2,067 | 2,442 |
   | Foot | 0,3025 | 0,2166 |
   | Hip | 0,2263 | 0,3424 |
   | IMU | 0,0069 | 0,011 |

   Các link còn lại cũng lệch. Tất cả đều lấy lại theo `inertial_report.csv`.

### Lỗi của script, đã sửa

3. Bảng `inertial_report` lúc 18:48 bị **ngược dấu ixy, ixz, iyz**, do tùy chọn "tensor notation" trong Mass Properties bị đổi (API của SolidWorks trả dấu theo tùy chọn đó).
   - Đã đổi dấu lại trong file báo cáo và kiểm tra khớp dấu với exporter ở Bubleft và IMU.
   - Script giờ dựng ma trận quán tính từ trục chính và mômen chính, nên không còn phụ thuộc tùy chọn này.
   - **Cách tính mới chưa chạy thử được** vì SolidWorks đã tắt. Lần tới mở SolidWorks, chạy lại một lần để xác nhận.

### Để ý khi làm sim

4. Limit khớp đang là giá trị tạm (±π, effort 10, velocity 1). Phải ghi đè trong code:
   - STS3095: khoảng 10,3 N·m, dùng cho Bub, Hip, Knee.
   - STS3215: khoảng 2,94 N·m, dùng cho rotate và Foot.
   - Velocity khoảng 4,7 rad/s.
   - Giới hạn góc thật chưa đo.
5. Tên joint không thống nhất: `Nhat_rotate_left`, `rotate_right`, `Kneeright` không có đuôi `_joint`. File `config/joint_names_OFFICIALdesign.yaml` còn có một tên rỗng ở đầu danh sách. Isaac không dùng file này, nhưng nếu dùng ROS thì phải sửa.
6. Trái/phải và dấu trục:
   - Tên trái/phải đang đặt theo góc nhìn người đứng đối diện robot. Link "left" nằm ở phía y âm của Baselink.
   - Trục Hip, Knee, Foot ngược dấu giữa hai bên.
   - Khớp Foot dùng hệ tọa độ mới nên góc 0 và trục khác bản URDF cũ (cũ là (0,0,−1), mới là (0,∓1,0)).
7. `Baselink.STL` nặng khoảng 15 MB, và exporter dùng luôn mesh hình ảnh làm mesh va chạm. Khi import nên dùng convex hull hoặc convex decomposition.
8. Đường dẫn mesh trong URDF có dạng `package://OFFICIALdesign/meshes/...`. Giữ nguyên tên thư mục `OFFICIALdesign` khi copy sang, nếu không importer có thể không tìm thấy mesh.

### Cần kiểm tra thêm (phải mở SolidWorks)

9. Trọng tâm của Hipleft và Hipright lệch nhau khoảng 8 mm theo trục z (−33 so với −25 mm), trong khi bản tính theo hình học của exporter lại đối xứng. Có thể một trong hai file `Hip+twist_105kg` / `Hip+twist_105kg(2)` có override khác nhau.
10. Trọng tâm Foot lệch sang ngang khoảng 18 mm, cùng dấu ở cả hai chân, trong khi exporter cho khoảng 1 mm. Hai chân dùng chung một file bàn chân, không có bản đối xứng. Cần xác nhận bàn chân thật có lắp giống hệt nhau không.

## Số liệu tổng quan (tư thế hiện tại trong CAD)

- Tổng khối lượng: **4,643 kg**.
- CoM cả robot trong hệ Baselink (x trước, y trái, z lên): (4,6; 4,4; −58,6) mm.
- Ở tư thế này robot đứng trên 2 bàn chân. CoM cao 340 mm so với mặt đất, nằm trong vùng tựa, cách mép khoảng 76 mm (xấp xỉ theo bounding box).
