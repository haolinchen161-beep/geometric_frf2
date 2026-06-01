"""
generate_data.py — 合成三维几何+FRF数据生成。

生成合成数据集用于验证 geometric_frf 训练流程。
- 3D 悬臂梁网格 (20×4×3 = 240 节点, 真三维坐标)
- 6-邻接边拓扑 (前后左右上下)
- 物理启发的 FRF (阻尼谐振子叠加, 3D空间模态)
- 输出: HDF5 文件 (train/val/test)

用法:
    F:\pytorch_cuda12\python.exe geometric_frf/sample/generate_data.py
"""

import h5py
import numpy as np
import os

# ============ 配置 ============
OUT_DIR = os.path.join(os.path.dirname(__file__), "data")
N_X, N_Y, N_Z = 20, 4, 3    # 3D网格: 长(x)×宽(y)×高(z)
N_POINTS = N_X * N_Y * N_Z  # 240 节点
L_X, L_Y, L_Z = 0.5, 0.04, 0.02  # 悬臂梁尺寸(m): 500×40×20mm
N_FREQS = 120
FREQ_MIN, FREQ_MAX = 1.0, 600.0  # 频率范围 (Hz), 600足够覆盖±10%尺寸变化下的第二峰
N_SAMPLES = 300
N_TRAIN, N_VAL, N_TEST = 200, 50, 50

DAMPING_ZETA = 0.003            # 铝材阻尼比 (固定值, 产生尖锐单峰)
AMPLITUDE_SCALE = 500000.0      # FRF 幅值缩放
NOISE_STD = 0.0001              # 极低噪声

# ============ 悬臂梁物理参数 (铝材) ============
E_ALUMINUM = 69e9               # 弹性模量 (Pa)
RHO_ALUMINUM = 2700             # 密度 (kg/m³)
E_VARIATION = 0.05              # 样本间弹模变化 ±5%
RHO_VARIATION = 0.03            # 样本间密度变化 ±3%

# 前3阶模态的 βL 值 (悬臂梁)
BETA_L = np.array([1.875104, 4.694091, 7.854757])

os.makedirs(OUT_DIR, exist_ok=True)


def create_3d_beam_mesh(nx=N_X, ny=N_Y, nz=N_Z, lx=L_X, ly=L_Y, lz=L_Z):
    """
    创建3D悬臂梁网格 (真三维坐标)。

    返回:
        points:    (N, 3) 节点坐标 [x, y, z] — 三个维度都有显著变化
        edges:     (2, E) 6-邻接边连接
        node_grid: (nz, ny, nx) → node_idx 映射
    """
    xs = np.linspace(0, lx, nx)
    ys = np.linspace(0, ly, ny)
    zs = np.linspace(0, lz, nz)

    points = []
    node_grid = np.zeros((nz, ny, nx), dtype=int)
    idx = 0
    for iz, z in enumerate(zs):
        for iy, y in enumerate(ys):
            for ix, x in enumerate(xs):
                points.append([x, y, z])
                node_grid[iz, iy, ix] = idx
                idx += 1

    points = np.array(points, dtype=np.float32)

    # 6-邻接边: 前/后, 左/右, 上/下
    edges = []
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                u = node_grid[iz, iy, ix]
                # x方向 (前/后)
                if ix + 1 < nx:
                    v = node_grid[iz, iy, ix + 1]
                    edges.extend([[u, v], [v, u]])
                # y方向 (左/右)
                if iy + 1 < ny:
                    v = node_grid[iz, iy + 1, ix]
                    edges.extend([[u, v], [v, u]])
                # z方向 (上/下)
                if iz + 1 < nz:
                    v = node_grid[iz + 1, iy, ix]
                    edges.extend([[u, v], [v, u]])

    edges = np.array(edges, dtype=np.int64).T  # (2, E)
    return points, edges, node_grid


def compute_beam_frf(frequencies, points, resonance_freqs, zeta, L, rng):
    """
    返回 (frf, omega, zeta_vals, phi):
      frf:      (N, F, 2) 复数FRF
      omega:    (K,) rad/s  固有圆频率
      zeta_vals:(K,) 阻尼比
      phi:      (N, K) 归一化模态振型
    """
    n_points = len(points)
    n_freqs = len(frequencies)
    n_modes = len(resonance_freqs)

    frf = np.zeros((n_points, n_freqs, 2), dtype=np.float32)
    omega_vals = np.zeros(n_modes, dtype=np.float32)
    zeta_vals = np.full(n_modes, zeta, dtype=np.float32)
    phi_all = np.zeros((n_points, n_modes), dtype=np.float32)
    x_coords = points[:, 0]
    omega = 2.0 * np.pi * frequencies

    for k in range(n_modes):
        omega_k = 2.0 * np.pi * resonance_freqs[k]
        beta = BETA_L[k] / L

        bx = beta * x_coords
        bL = BETA_L[k]
        sigma = (np.cosh(bL) + np.cos(bL)) / (np.sinh(bL) + np.sin(bL))
        phi = (np.cosh(bx) - np.cos(bx) - sigma * (np.sinh(bx) - np.sin(bx)))
        phi = phi / (np.abs(phi).max() + 1e-10)

        dw = omega_k**2 - omega**2
        gamma = 2.0 * zeta * omega_k * omega
        D = np.maximum(dw**2 + gamma**2, 1e-10)
        frf[..., 0] += np.outer(phi, AMPLITUDE_SCALE * dw / D)
        frf[..., 1] += np.outer(phi, -AMPLITUDE_SCALE * gamma / D)

        omega_vals[k] = omega_k
        phi_all[:, k] = phi

    return frf, omega_vals, zeta_vals, phi_all


def add_point_features_3d(points, rng, lx, ly, lz):
    """
    生成逐点特征: 材料属性空间变化 (3D)。

    返回:
        point_features: (N, 3) [密度偏差, 弹性模量偏差, 截面惯性矩偏差]
    """
    N = len(points)
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    # 悬臂梁: 固定端在 x=0, 自由端在 x=lx
    density = 1.0 + rng.normal(0, 0.02, N)
    modulus = 1.0 - 0.05 * (x / lx) + rng.normal(0, 0.01, N)
    section = 1.0 - 0.1 * np.abs(y - ly/2) / (ly/2) * np.abs(z - lz/2) / (lz/2)

    return np.column_stack([density, modulus, section]).astype(np.float32)


def compute_beam_natural_frequencies(E, rho, L, b, h):
    """Euler-Bernoulli 悬臂梁固有频率 (Hz)"""
    A = b * h
    I = b * h**3 / 12.0
    EI = E * I
    rhoA = rho * A
    # f_n = (β_n·L)² / (2π·L²) · sqrt(EI/ρA)
    return (BETA_L**2) / (2.0 * np.pi * L**2) * np.sqrt(EI / rhoA)


def generate_sample(rng, sample_idx):
    """
    生成单个 3D 悬臂梁样本 (Euler-Bernoulli 物理模型)。

    仅含 [1, 500] Hz 内的固有频率 (通常 2 个峰值)。
    其余频率区间 FRF 为平滑非共振响应。
    """
    # 样本间几何变化: 梁尺寸随机 ±10%, 网格分辨率不变
    lx = L_X * rng.uniform(0.9, 1.1)
    ly = L_Y * rng.uniform(0.9, 1.1)
    lz = L_Z * rng.uniform(0.9, 1.1)
    points, edges, node_grid = create_3d_beam_mesh(lx=lx, ly=ly, lz=lz)

    # 微小几何缺陷 (固定端节点不加噪声, 保证x=0边界条件)
    noise = rng.normal(0, 0.001, points.shape)  # 1mm 量级
    noise[points[:, 0] == 0.0] = 0.0
    points += noise

    # 样本间材料参数微变 (先于频率网格: 需要共振频率确定峰值位置)
    E = E_ALUMINUM * (1.0 + rng.uniform(-E_VARIATION, E_VARIATION))
    rho = RHO_ALUMINUM * (1.0 + rng.uniform(-RHO_VARIATION, RHO_VARIATION))

    # 计算该样本的固有频率 (使用变化的几何尺寸)
    all_freqs = compute_beam_natural_frequencies(E, rho, lx, ly, lz)
    mask = (all_freqs >= FREQ_MIN) & (all_freqs <= FREQ_MAX)
    resonance_freqs = all_freqs[mask]  # 通常 2 个: ~65 Hz, ~409 Hz
    n_modes = len(resonance_freqs)

    # 自适应频率网格: 每个样本在自身共振峰周围 ±3·半功率带宽内密集线性采样
    # 确保训练 target 精确捕获每个样本的峰值中心
    resonance_freqs_sorted = np.sort(resonance_freqs)
    N_PER_PEAK = 35  # 每个共振峰的线性采样点数

    segments = []
    prev = FREQ_MIN
    for f_k in resonance_freqs_sorted:
        bw = 2.0 * DAMPING_ZETA * f_k
        pk_lo = max(FREQ_MIN, f_k - 3.0 * bw)
        pk_hi = min(FREQ_MAX, f_k + 3.0 * bw)
        if prev < pk_lo:
            segments.append(('log', prev, pk_lo))
        segments.append(('linear', pk_lo, pk_hi))
        prev = pk_hi
    if prev < FREQ_MAX:
        segments.append(('log', prev, FREQ_MAX))

    # 精确分配点数: 线性段各 N_PER_PEAK, 对数段按跨度比例分配
    total_log_span = sum(hi - lo for t, lo, hi in segments if t == 'log') + 1e-10
    n_log_total = N_FREQS - N_PER_PEAK * n_modes
    alloc = []
    for t, lo, hi in segments:
        if t == 'linear':
            alloc.append(N_PER_PEAK)
        else:
            alloc.append(max(2, round(n_log_total * (hi - lo) / total_log_span)))
    # 修正舍入误差使总数恰好 N_FREQS
    diff = N_FREQS - sum(alloc)
    log_idxs = [i for i, (t, _, _) in enumerate(segments) if t == 'log']
    log_idxs.sort(key=lambda i: segments[i][2] - segments[i][1], reverse=True)
    for j in range(abs(diff)):
        alloc[log_idxs[j % len(log_idxs)]] += 1 if diff > 0 else -1

    freq_parts = []
    for (t, lo, hi), n_pts in zip(segments, alloc):
        if t == 'linear':
            freq_parts.append(np.linspace(lo, hi, n_pts, endpoint=True))
        else:
            freq_parts.append(np.logspace(np.log10(lo), np.log10(hi), n_pts, endpoint=False))
    frequencies = np.concatenate(freq_parts).astype(np.float32)

    # 生成 FRF (Euler-Bernoulli 梁理论)
    point_frf, modal_omega, modal_zeta, modal_phi = compute_beam_frf(
        frequencies, points, resonance_freqs, DAMPING_ZETA, lx, rng)
    point_frf += rng.normal(0, NOISE_STD, point_frf.shape)

    # 逐点材料特征
    point_features = add_point_features_3d(points, rng, lx, ly, lz)

    # 全局特征: E, rho, 共振频率
    global_feat = np.zeros(8, dtype=np.float32)
    global_feat[0] = E / E_ALUMINUM       # 归一化弹性模量
    global_feat[1] = rho / RHO_ALUMINUM    # 归一化密度
    global_feat[2] = n_modes               # 模态数
    for k in range(min(n_modes, 5)):
        global_feat[3 + k] = resonance_freqs[k] / FREQ_MAX

    return (points.astype(np.float32),
            edges.astype(np.int64),
            point_features.astype(np.float32),
            point_frf.astype(np.float32),
            frequencies.astype(np.float32),
            global_feat.astype(np.float32),
            modal_omega.astype(np.float32),
            modal_zeta.astype(np.float32),
            modal_phi.astype(np.float32))


def save_hdf5(filepath, samples):
    """将样本列表保存为 HDF5 文件"""
    n_samples = len(samples)

    all_points = np.stack([s[0] for s in samples])
    all_edges = np.stack([s[1] for s in samples])
    all_features = np.stack([s[2] for s in samples])
    all_frf = np.stack([s[3] for s in samples])
    all_freqs = np.stack([s[4] for s in samples])
    all_global = np.stack([s[5] for s in samples])
    all_omega = np.stack([s[6] for s in samples])
    all_zeta  = np.stack([s[7] for s in samples])
    all_phi   = np.stack([s[8] for s in samples])

    with h5py.File(filepath, 'w') as f:
        f.create_dataset('points', data=all_points)
        f.create_dataset('edges', data=all_edges)
        f.create_dataset('point_features', data=all_features)
        f.create_dataset('point_frf', data=all_frf)
        f.create_dataset('frequencies', data=all_freqs)
        f.create_dataset('phy_para', data=all_global)
        f.create_dataset('modal_omega', data=all_omega)
        f.create_dataset('modal_zeta', data=all_zeta)
        f.create_dataset('modal_phi', data=all_phi)

    print(f"  保存: {filepath}")
    print(f"    points:       {all_points.shape}")
    print(f"    point_frf:    {all_frf.shape}")
    print(f"    modal_omega:  {all_omega.shape}")
    print(f"    modal_phi:    {all_phi.shape}")


def main():
    print("=" * 60)
    print("生成合成 FRF 数据集")
    print("=" * 60)
    print(f"3D网格: {N_X}×{N_Y}×{N_Z} = {N_POINTS} 节点 (真三维)")
    print(f"尺寸: {L_X}×{L_Y}×{L_Z} m (悬臂梁)")
    print(f"频率: {N_FREQS} 点, [{FREQ_MIN}, {FREQ_MAX}] Hz")
    print(f"样本: {N_SAMPLES} (训练{N_TRAIN}, 验证{N_VAL}, 测试{N_TEST})")
    print()

    # 固定种子确保可复现
    master_rng = np.random.RandomState(42)

    # 生成所有样本
    print("生成数据...")
    all_samples = []
    for i in range(N_SAMPLES):
        rng = np.random.RandomState(42 + i)
        sample = generate_sample(rng, i)
        all_samples.append(sample)
        if (i + 1) % 50 == 0:
            print(f"  已生成 {i+1}/{N_SAMPLES}")

    # 划分
    train_samples = all_samples[:N_TRAIN]
    val_samples = all_samples[N_TRAIN:N_TRAIN + N_VAL]
    test_samples = all_samples[N_TRAIN + N_VAL:]

    print()
    print("保存 HDF5 文件...")
    save_hdf5(os.path.join(OUT_DIR, "train.h5"), train_samples)
    save_hdf5(os.path.join(OUT_DIR, "val.h5"), val_samples)
    save_hdf5(os.path.join(OUT_DIR, "test.h5"), test_samples)

    print()
    print("完成! 数据文件位于:", OUT_DIR)
    print(f"  {os.path.join(OUT_DIR, 'train.h5')}")
    print(f"  {os.path.join(OUT_DIR, 'val.h5')}")
    print(f"  {os.path.join(OUT_DIR, 'test.h5')}")


if __name__ == '__main__':
    main()
