# newsimple — bản 6 DOF

SW2URDF, **2026-03-17 23:10**. Hip, Knee, Foot × 2. Không có Bub, không có Twist.

Là asset của task 6DOF cũ `NewSimple-Walk-v0` (đã comment out trong `__init__.py`)
— checkpoint `model_349.pt` train trên bản này. Giữ lại để rollback / đối chiếu.

| Giới hạn cơ khí | Hip | Knee | Foot |
|---|---|---|---|
| `usd/NewSimple.usd` | ±37.30 | ±126.05 | −65.21 / 40.26 |

`usd/NewSimple.usd` là vỏ bọc, tham chiếu `./NewSimple/NewSimple.usd`.

Mesh **không** dùng chung với `fulltrans/` — trùng tên nhưng khác file
(`Baselink.STL` ở đây 7.1 MB, bên `fulltrans/` 6.6 MB, md5 khác).
