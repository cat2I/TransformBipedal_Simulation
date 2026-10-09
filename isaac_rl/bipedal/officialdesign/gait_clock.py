"""Hệ số gait clock theo cách dựng pha của apex phase_function.py.

Pha chuẩn hóa p thuộc [0, 1); thứ tự hệ số: [r_frc, r_vel, l_frc, l_vel].
Dấu hệ số: -1 phạt, 0 bỏ qua, +1 thưởng đại lượng tương ứng.

Dùng scipy PCHIP lúc khởi tạo để dựng bảng tra (LUT), sau đó đưa bảng
sang torch trên thiết bị chạy mô phỏng. Runtime chỉ tra bảng để xử lý
đồng thời nhiều env, không gọi scipy và không import code gaitclockref.
"""

# gait_clock.py mô tả lịch chân: tại pha đó, lực và tốc độ của mỗi chân nhận hệ số nào.
import numpy as np
import torch

#hàm này trả về hệ số thưởng phạt lực và vận tốc theo từng pha 
def _phase_coefficients(
    stance_mode: str,
    have_incentive: bool,
) -> np.ndarray:
    """Trả hệ số (4 pha, 4 đại lượng), trước khi làm mượt chuyển pha.

    Hàng: phải vung, chống kép thứ nhất, trái vung, chống kép thứ hai.
    Cột: lực phải, tốc độ phải, lực trái, tốc độ trái.

    stance_mode chỉ quy định hai khoảng giữa các pha vung:
    grounded = hai chân chống; aerial = hai chân bay; zero = bỏ qua.
    """
    if stance_mode not in ("grounded", "aerial", "zero"): #các pha vung, cùng trên không hoặc 0 làm gì 
        raise ValueError(
            "stance_mode must be 'grounded', 'aerial', or 'zero'"
        )

    incentive = 1.0 if have_incentive else 0.0 #nếu have_incen = true thì gán = 1, false thì 0 

    right_swing = [-1.0, incentive, incentive, -1.0] #hệ số phạt khi chân phải vung 
    left_swing = [incentive, -1.0, -1.0, incentive] #hệ số phạt khi chân trái vung 

    if stance_mode == "grounded":
        between_swings = [incentive, -1.0, incentive, -1.0] #hệ số phạt khi hai chân chống 
    elif stance_mode == "aerial":
        between_swings = [-1.0, incentive, -1.0, incentive] #hệ số phạt khi hai chân bay 
    else:
        between_swings = [0.0, 0.0, 0.0, 0.0] #hệ số phạt khi bỏ qua 

    return np.array(
        [ #trả về bộ hệ số thưởng phạt lực và vận tốc của chân trái và phải theo từng pha 
            right_swing,
            between_swings,
            left_swing,
            between_swings,
        ],
        dtype=np.float64,
    )

# chia 8 mốc giữa 4 pha để làm các mốc cho quá trình chuyển tiếp 
def _phase_transition_points(swing_ratio: float, strict_relaxer: float) -> np.ndarray: #hàm trả về mảng numpy 
    """Trả 8 mốc pha tăng nghiêm ngặt để dựng PCHIP.

    Mỗi pha có hai mốc nằm lùi vào trong một khoảng bằng
    strict_relaxer nhân độ dài pha, giống cách dựng của apex.
    """
    if not 0.0 < swing_ratio < 0.5:
        raise ValueError("swing_ratio must be in (0, 0.5)")
    if not 0.0 < strict_relaxer < 0.5:
        raise ValueError("strict_relaxer must be in (0, 0.5)")

    boundaries = np.array(
        [0.0, swing_ratio, 0.5, 0.5 + swing_ratio, 1.0],
        dtype=np.float64,
    )# chia 0 -> 1 thành 4 khoảng, mỗi khoảng là 1 pha, lấy 2 mốc lùi vào trong mỗi pha để làm mốc chuyển tiếp

    # lấy điểm đầu, điểm cuối từng pha -> tính khoảng lùi vào trong mỗi pha để làm mốc chuyển tiếp
    starts = boundaries[:-1] #điểm đầu từng pha (bỏ đi điểm cuối)
    ends = boundaries[1:] #điểm cuối từng pha (bỏ điểm đầu)

    # tính khoảng lùi theo độ dài từng pha 
    offsets = (ends - starts) * strict_relaxer

    return np.stack(
        (starts + offsets, ends - offsets), # khoảng offset = điểm cuối - offset và start - offset 
        axis=1,
    ).reshape(-1) #xếp thành mảng 1 chiều 

# ghép mốc pha với hệ số và nối 3 chu kì -> làm mượt đoạn chuyển pha thay vì  
def _periodic_phase_coefficients(swing_ratio: float, strict_relaxer: float, stance_mode: str, have_incentive: bool) -> tuple[np.ndarray, np.ndarray]:
    
    """Trả mốc (24,) và hệ số (24, 4) của ba chu kỳ liên tiếp.

    Nối chu kỳ trước, hiện tại và sau để PCHIP nội suy qua
    ranh giới p=0 và p=1 bằng cùng quy luật chuyển pha.
    """
    knots = _phase_transition_points(swing_ratio, strict_relaxer)
    coefficients = _phase_coefficients(stance_mode, have_incentive)

    # Mỗi pha có hai mốc nhận cùng một bộ hệ số: (4, 4) -> (8, 4).
    values = np.repeat(coefficients, repeats=2, axis=0)

    # Trục pha chuẩn hóa: một chu kỳ dài đúng 1.0.
    periodic_knots = np.concatenate(
        (knots - 1.0, knots, knots + 1.0) #cùng các mốc nhưng của chu kì trước và chu kì sau 
    )
    periodic_values = np.concatenate( #hàm này nối 3 mảng thành 1 mảng dài 24 số: chu kì trc - chu kì hiện tại - chu kì sau 
        (values, values, values),
        axis=0,
    )

    return periodic_knots, periodic_values

# hàm này build gait clock 
def build_gait_clock(swing_ratio: float = 0.35,
    strict_relaxer: float = 0.1,
    stance_mode: str = "grounded",
    have_incentive: bool = False,
    num_samples: int = 4096,
    device: str | torch.device = "cpu", # mặc định dùng CPU
) -> torch.Tensor:
    """Dựng LUT (4, N), thứ tự [r_frc, r_vel, l_frc, l_vel].

    Cột j ứng với pha p = j / N, không chứa điểm p = 1.
    N phải chẵn để nửa chu kỳ tương ứng đúng N/2 cột.

    PCHIP chỉ chạy trên CPU lúc dựng bảng; bảng kết quả được chuyển
    sang device để runtime tra bằng Torch.
    """
    from scipy.interpolate import PchipInterpolator
    # kiểm tra các tham số num_samples có phải số nguyên chẵn > 2 ko 
    #
    if not isinstance(num_samples, int) or isinstance(num_samples, bool):
        raise ValueError("num_samples must be an integer")
    if num_samples < 2 or num_samples % 2 != 0:
        raise ValueError("num_samples must be even and >= 2")

    # hàm trả về knots, value dựa trên hàm phase_transition_points ở trên 
    knots, values = _periodic_phase_coefficients(
        swing_ratio,
        strict_relaxer,
        stance_mode,
        have_incentive,
    )

    # nội suy dọc theo 24 mốc mỗi pha 
    spline = PchipInterpolator(
        knots,
        values,
        axis=0,
        extrapolate=False,
    )

    # 
    phases = np.arange(num_samples, dtype=np.float64) / num_samples
    samples = spline(phases)  # (N, 4) : N hàng, 4 cột: N vị trí pha 

    # Mỗi hàng là một đường hệ số; mỗi cột là một vị trí pha.
    table = torch.tensor(
        samples.T,
        dtype=torch.float32,
        device=device,
    )
    return table.contiguous()


#hàm tra bảng theo pha (giống kiểu bảng cửu chương )
def lookup_gait_clock(
    table: torch.Tensor,
    phase: torch.Tensor,
) -> torch.Tensor:
    """Tra LUT (4, N) bằng pha (num_envs,), trả hệ số (num_envs, 4).

    Table và phase phải ở cùng thiết bị.
    Wrap pha về [0, 1), rồi nội suy tuyến tính giữa hai cột gần nhất.
    Thứ tự đầu ra: [r_frc, r_vel, l_frc, l_vel].
    """
    num_samples = table.shape[1]

    position = torch.remainder(phase, 1.0) * num_samples
    lower = torch.floor(position)
    weight = (position - lower).unsqueeze(-1)

    index0 = lower.to(dtype=torch.int64) % num_samples
    index1 = (index0 + 1) % num_samples

    values0 = table[:, index0].transpose(0, 1)
    values1 = table[:, index1].transpose(0, 1)

    return values0 + weight * (values1 - values0)