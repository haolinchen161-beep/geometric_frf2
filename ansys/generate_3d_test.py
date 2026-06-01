"""
ANSYS 悬臂板数据集生成 — 模态参数 + FRF。
Sobol采样 + 1D距离特征 + 全宽悬臂夹具。

物理正确的位移频响函数:
    H(x, x_f, ω) = Σ_k φ_k(x)·φ_k(x_f) / (ω_k² - ω² + j·2ζ_k·ω_k·ω)
"""
from ansys.mapdl.core import launch_mapdl
from scipy.stats import qmc
import numpy as np
import h5py
import os
import time

# ============ 配置 ============
N_SAMPLES = 300
N_TRAIN, N_VAL, N_TEST = 200, 50, 50
N_MODES = 2
N_FREQS = 40
FREQ_MIN, FREQ_MAX = 1.0, 5000.0
MESH_SIZE = 0.006
CLAMP_DEPTH = 0.008           # 夹持深度 8mm
DAMPING_ZETA = 0.003
AMPLITUDE_SCALE = 500000.0
OUT_DIR = os.path.join(os.path.dirname(__file__), "data")
VIZ_DIR  = os.path.join(os.path.dirname(__file__), "mesh_viz")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(VIZ_DIR, exist_ok=True)

# 基准物理参数 (铝材) — 悬臂板
E_BASE, RHO_BASE = 71.7e9, 2810.0
L_BASE, W_BASE, H_BASE = 0.100, 0.060, 0.010  # 100×60×10mm

print(">>> 正在生成 Sobol 低偏差序列...")
sampler = qmc.Sobol(d=5, scramble=True, seed=42)
sobol_samples = sampler.random(n=N_SAMPLES)
# E(±5%), rho(±3%), L/W/H(±10%)
l_bounds = [0.95, 0.97, 0.90, 0.90, 0.90]
u_bounds = [1.05, 1.03, 1.10, 1.10, 1.10]
scaled_sobol = qmc.scale(sobol_samples, l_bounds, u_bounds)

print(f"配置: {N_SAMPLES}样本(Sobol), {N_MODES}阶模态, {N_FREQS}频率点, 悬臂{CLAMP_DEPTH*1000:.0f}mm, 网格{MESH_SIZE*1000:.0f}mm")
print(f"板: {L_BASE*1000:.0f}×{W_BASE*1000:.0f}×{H_BASE*1000:.0f}mm")

print(">>> 正在连接 ANSYS 求解器...")
mapdl = launch_mapdl(override=True)
print(f">>> 连接成功! 版本: {mapdl.version}\n")

# 预分配
all_points, all_frf, all_freqs = [], [], []
all_omega, all_zeta, all_phi, all_phi_exc = [], [], [], []
all_features = []
t0 = time.time()

for i in range(N_SAMPLES):
    print(f"[{i+1}/{N_SAMPLES}]", end=" ", flush=True)
    try:
        mapdl.clear()
        mapdl.prep7()
    except Exception:
        print("(reconnect)", end=" ", flush=True)
        mapdl.exit(); time.sleep(2)
        mapdl = launch_mapdl(override=True)
        mapdl.clear(); mapdl.prep7()

    # 1. Sobol 参数
    E = E_BASE * scaled_sobol[i, 0]
    rho = RHO_BASE * scaled_sobol[i, 1]
    L = L_BASE * scaled_sobol[i, 2]
    W = W_BASE * scaled_sobol[i, 3]
    H = H_BASE * scaled_sobol[i, 4]

    # 2. 建模 + 网格
    mapdl.mp("EX", 1, E)
    mapdl.mp("PRXY", 1, 0.33)
    mapdl.mp("DENS", 1, rho)
    mapdl.block(0, L, 0, W, 0, H)
    mapdl.et(1, "SOLID187")
    mapdl.mshape(1, "3D")
    mapdl.mshkey(0)
    mapdl.esize(MESH_SIZE)
    try:
        mapdl.vmesh("ALL")
    except Exception:
        mapdl.smrtsize(4)
        mapdl.vmesh("ALL")

    # 3. 悬臂夹具: 全宽夹紧 x∈[0, 8mm]
    mapdl.nsel("S", "LOC", "X", 0, CLAMP_DEPTH)
    mapdl.d("ALL", "ALL")
    mapdl.allsel()

    # 4. 模态分析
    mapdl.slashsolu()
    mapdl.antype("MODAL")
    mapdl.modopt("LANB", N_MODES, "ON")
    mapdl.solve()

    # 5. 提取结果
    mapdl.post1()
    coords = np.array(mapdl.mesh.nodes, dtype=np.float32)
    n_nodes = len(coords)

    omega_k = np.zeros(N_MODES, dtype=np.float32)
    phi_z = np.zeros((n_nodes, N_MODES), dtype=np.float32)

    for k in range(1, N_MODES + 1):
        mapdl.set(1, k)
        f_hz = mapdl.post_processing.freq
        omega_k[k-1] = 2.0 * np.pi * f_hz
        disp = np.array(mapdl.post_processing.nodal_displacement("ALL"), dtype=np.float32)
        phi_z[:, k-1] = disp[:, 2]

    # 6. 激励点: 自由端最远角 (L, W, H)
    exc_x, exc_y, exc_z = L, W, H
    dist = np.sqrt((coords[:, 0] - exc_x)**2 + (coords[:, 1] - exc_y)**2 + (coords[:, 2] - exc_z)**2)
    exc_idx = np.argmin(dist)
    phi_exc_k = phi_z[exc_idx, :].copy()

    zeta_k = np.full(N_MODES, DAMPING_ZETA, dtype=np.float32)

    # 7. 自适应频率网格
    freqs_parts = []
    prev = FREQ_MIN
    for f_k in omega_k / (2*np.pi):
        bw = 2.0 * DAMPING_ZETA * f_k
        lo = max(FREQ_MIN, f_k - 3.0 * bw)
        hi = min(FREQ_MAX, f_k + 3.0 * bw)
        if prev < lo:
            freqs_parts.append(np.logspace(np.log10(max(prev, 0.1)), np.log10(lo),
                                max(2, int(5 * (lo-prev)/FREQ_MAX)), endpoint=False))
        freqs_parts.append(np.linspace(lo, hi, max(8, int(12 * (hi-lo)/FREQ_MAX)), endpoint=True))
        prev = hi
    if prev < FREQ_MAX:
        freqs_parts.append(np.logspace(np.log10(max(prev, 0.1)), np.log10(FREQ_MAX),
                            max(2, int(5 * (FREQ_MAX-prev)/FREQ_MAX)), endpoint=True))
    freqs = np.unique(np.sort(np.concatenate(freqs_parts)))
    if len(freqs) > N_FREQS:
        idx = np.linspace(0, len(freqs)-1, N_FREQS, dtype=int)
        freqs = freqs[idx]
    elif len(freqs) < N_FREQS:
        freqs = np.interp(np.linspace(0, 1, N_FREQS),
                          np.linspace(0, 1, len(freqs)), freqs)
    freqs = freqs.astype(np.float32)

    # 8. FRF
    omega_q = 2.0 * np.pi * freqs
    frf = np.zeros((n_nodes, len(freqs), 2), dtype=np.float32)
    for k in range(N_MODES):
        wk = omega_k[k]; zk = zeta_k[k]
        pk = phi_z[:, k] * phi_exc_k[k]
        dw = wk**2 - omega_q**2
        gm = 2.0 * zk * wk * omega_q
        D = np.maximum(dw**2 + gm**2, 1e-10)
        frf[:, :, 0] += np.outer(pk, AMPLITUDE_SCALE * dw / D)
        frf[:, :, 1] += np.outer(pk, -AMPLITUDE_SCALE * gm / D)

    # 9. 全局特征 (6维) — SIREN从(x,y,z)自己学空间分布
    gf = np.array([E/E_BASE, rho/RHO_BASE, L/L_BASE, W/W_BASE, H/H_BASE, N_MODES], dtype=np.float32)

    all_points.append(coords)
    all_frf.append(frf)
    all_freqs.append(freqs)
    all_omega.append(omega_k)
    all_zeta.append(zeta_k)
    all_phi.append(phi_z)
    all_phi_exc.append(phi_exc_k)
    all_features.append(gf)

    exc_actual = coords[exc_idx]
    print(f"N={n_nodes}, f=[{omega_k[0]/(2*np.pi):.0f}~{omega_k[-1]/(2*np.pi):.0f}]Hz, "
          f"EXC=({exc_actual[0]*1000:.0f},{exc_actual[1]*1000:.0f})mm, F={len(freqs)}")
    time.sleep(0.5)  # 防止MAPDL过载断连

    # 10. 网格可视化 (前10个)
    if i < 10:
        try:
            import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
            import pyvista as pv
            mesh = mapdl.mesh._grid
            plotter = pv.Plotter(off_screen=True, window_size=[800, 600])
            plotter.add_mesh(mesh, color='lightblue', show_edges=True, edge_color='gray', line_width=0.3)
            plotter.add_text(f'Sample {i+1}: {n_nodes} nodes, f1={omega_k[0]/(2*np.pi):.0f}Hz', font_size=10)
            plotter.camera_position = 'iso'
            plotter.screenshot(os.path.join(VIZ_DIR, f'sample_{i:03d}_mesh.png'))
            plotter.close()
        except Exception:
            pass

mapdl.exit()
elapsed = time.time() - t0
print(f"\n生成完成, 耗时 {elapsed:.0f}s")

# 保存 HDF5
def save_h5(name, idx_slice):
    idxs = list(idx_slice)
    with h5py.File(os.path.join(OUT_DIR, name), 'w') as f:
        for i, idx in enumerate(idxs):
            grp = f.create_group(f'sample_{i}')
            grp.create_dataset('points', data=all_points[idx])
            grp.create_dataset('point_frf', data=all_frf[idx])
            grp.create_dataset('frequencies', data=all_freqs[idx])
            grp.create_dataset('modal_omega', data=all_omega[idx])
            grp.create_dataset('modal_zeta', data=all_zeta[idx])
            grp.create_dataset('modal_phi', data=all_phi[idx])
            grp.create_dataset('modal_phi_exc', data=all_phi_exc[idx])
            grp.create_dataset('point_features', data=all_features[idx])
    print(f"  保存: {name} ({len(idxs)}样本)")

save_h5('train.h5', range(N_TRAIN))
save_h5('val.h5', range(N_TRAIN, N_TRAIN+N_VAL))
save_h5('test.h5', range(N_TRAIN+N_VAL, N_SAMPLES))

# FRF可视化 (第1个样本)
print("\n生成FRF可视化...")
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
coords0 = all_points[0]; frf0 = all_frf[0]; freqs0 = all_freqs[0]
amp0 = np.sqrt(frf0[..., 0]**2 + frf0[..., 1]**2)
n_nodes0 = len(coords0)

# 沿x方向选5个点 (悬臂: 固支端→自由端)
x_sorted = np.argsort(coords0[:, 0])
n_pts = n_nodes0
idx_list = [x_sorted[0], x_sorted[n_pts//4], x_sorted[n_pts//2],
            x_sorted[3*n_pts//4], x_sorted[-1]]
labels = ['固定端', 'x=1/4', 'x=1/2', 'x=3/4', '自由端']

fig, axes = plt.subplots(5, 1, figsize=(14, 16), sharex=True)
for ax, idx, label in zip(axes, idx_list, labels):
    ax.semilogx(freqs0, amp0[idx], 'b-', linewidth=1.2)
    ax.set_ylabel(f'{label}\n(x={coords0[idx,0]:.3f})')
    ax.grid(alpha=0.3); ax.set_ylim(0, amp0[idx].max()*1.1)
axes[-1].set_xlabel('Frequency (Hz)')
fig.suptitle(f'Cantilever Plate FRF — {n_nodes0} nodes, f1={all_omega[0][0]/(2*np.pi):.0f}Hz', fontsize=14)
plt.tight_layout(); plt.savefig(os.path.join(VIZ_DIR, 'sample_000_frf.png'), dpi=150); plt.close()
print(f"可视化保存: {VIZ_DIR}/")
