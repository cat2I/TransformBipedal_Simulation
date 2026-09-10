# trans3dof — bản đơn giản hoá

SW2URDF, **2026-02-20 06:05**. 6 khớp (Hip, Knee, Foot × 2) nhưng **hình học khác hẳn**
`newsimple/`, và giới hạn khớp **bất đối xứng**:

| Giới hạn cơ khí | Hip | Knee | Foot |
|---|---|---|---|
| `usd/3DOFTrans.usd` | −21.26 / 36.39 | −84.98 / 86.67 | −40.26 / 65.21 |

So với `newsimple/` (Hip ±37.30 đối xứng) — đây là hai đời robot khác nhau, không phải
hai cách cấu hình cùng một robot.

`usd/3DOFTrans.usd` là vỏ bọc, tham chiếu `./3DOFTrans/3DOFTrans.usd`.
