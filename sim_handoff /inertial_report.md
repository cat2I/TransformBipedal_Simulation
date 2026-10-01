# Inertial parameters — OFFICIALdesign

Xuat ngay 2026-09-30 tu SolidWorks (co ton trong Override Mass).
Moi link: CoM va inertia bieu dien trong he toa do cua link (= coordinate system cua joint cha trong SW),
inertia lay tai CoM, don vi kg·m². Tu the: tu the hien tai trong CAD (goc khop = 0).

**Tong khoi luong robot:** 4.6430 kg  
**CoM toan robot trong he Baselink (x truoc, y trai, z len):** (4.6, 4.4, -58.6) mm

| Link | Frame | Mass (kg) | CoM x,y,z (mm) | ixx | iyy | izz | ixy | ixz | iyz |
|---|---|---|---|---|---|---|---|---|---|
| Baselink | Baselink | 2.4416 | 6.3, 5.1, 32.3 | 2.869e-02 | 1.972e-02 | 4.496e-02 | -3.862e-04 | 8.807e-05 | 1.076e-04 |
| IMUleft | imu left | 0.0110 | 0.2, 0.0, -2.1 | 1.310e-06 | 9.034e-07 | 1.930e-06 | -7.408e-08 | -4.076e-09 | -8.831e-11 |
| IMUright | imuright | 0.0110 | 0.2, 0.0, -2.1 | 1.310e-06 | 9.034e-07 | 1.930e-06 | -7.408e-08 | -4.076e-09 | -8.831e-11 |
| Bubleft | Bubleft | 0.0768 | -17.0, -0.7, 0.0 | 6.340e-05 | 3.223e-05 | 7.305e-05 | -1.939e-06 | 2.288e-11 | -1.959e-11 |
| Hipleft | Hipleft | 0.3424 | -0.2, 2.8, -33.0 | 4.234e-04 | 3.525e-04 | 1.284e-04 | 1.134e-07 | 4.663e-07 | 2.592e-05 |
| Rotate_left | Nhat_rotate_left | 0.1058 | -4.0, -0.0, -34.6 | 9.982e-05 | 1.502e-04 | 8.967e-05 | 3.069e-11 | 3.579e-06 | -8.936e-12 |
| Kneeleft | Kneeleft | 0.3481 | -0.1, 0.1, -65.5 | 1.695e-03 | 1.620e-03 | 1.414e-04 | -2.052e-08 | -2.234e-06 | -1.352e-05 |
| Footleft | Footleft | 0.2166 | 40.0, 18.3, -41.7 | 2.774e-04 | 3.871e-04 | 4.790e-04 | -5.669e-05 | -2.309e-07 | 6.608e-05 |
| Bubright | Bubright | 0.0768 | -17.0, 0.7, -0.0 | 6.340e-05 | 3.223e-05 | 7.305e-05 | 1.939e-06 | -2.288e-11 | -1.959e-11 |
| Hipright | Hipright | 0.3424 | -3.2, -2.7, -25.2 | 3.887e-04 | 3.375e-04 | 1.484e-04 | 2.490e-06 | 1.905e-05 | -3.176e-05 |
| Rotate_right | Nhat_rotate_right | 0.1058 | 4.0, 0.0, -34.6 | 9.982e-05 | 1.502e-04 | 8.967e-05 | 3.069e-11 | -3.579e-06 | 8.936e-12 |
| Kneeright | Kneeright | 0.3481 | 0.1, -0.2, -65.5 | 1.695e-03 | 1.620e-03 | 1.413e-04 | -1.431e-08 | 2.234e-06 | 1.044e-05 |
| Footright | Footright | 0.2166 | 40.0, 19.5, -41.6 | 2.867e-04 | 3.878e-04 | 4.877e-04 | -5.961e-05 | 6.146e-08 | 6.991e-05 |

## On dinh tinh (tu the hien tai trong CAD)

- Component cham dat: foot 1^OFFICIALdesign-1, foot 2^OFFICIALdesign-1
- CoM chieu xuong dat (he Baselink): x = 4.6 mm, y = 4.4 mm
- Do cao CoM so voi mat dat: 340.1 mm
- Vung tua (bao loi, xap xi bang bounding box, mm): (-72, -123), (100, -123), (101, 29), (101, 123), (-71, 123), (-72, -29)
- CoM NAM TRONG vung tua; khoang cach toi mep gan nhat: 76.4 mm

Luu y: vung tua lay tu bounding box nen rong hon vet tiep xuc that; tu the khac (che do banh xe,
luc bien hinh) phai dat robot vao tu the do trong CAD roi chay lai script.
