"""
查看复数FRF数据的真实值：幅值曲线 + 峰值放大 + Re/Im平滑过零。
用法: F:\pytorch_cuda12\python.exe geometric_frf/sample/测试.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import torch, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from geometric_frf.data.dataset import GeometricHDF5Dataset

data_dir = os.path.join(os.path.dirname(__file__), "data")
out_dir  = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(out_dir, exist_ok=True)

# 加载原始数据 (不做归一化)
CONFIG = {'freq_sample': False, 'freq_limit': 90, 'n_freqs': 120}
ds = GeometricHDF5Dataset(['train.h5'], CONFIG, data_dir=data_dir, normalization=False, test=True)
sample = ds[0]

points_3d = sample['geometry'].points       # (240, 3)
freq_raw  = sample['frequencies']           # (120,) Hz
frf_cmplx = sample['point_frf']             # (240, 120, 2) [Re, Im]
frf_amp   = torch.sqrt(frf_cmplx[..., 0]**2 + frf_cmplx[..., 1]**2 + 1e-8)

f_np = freq_raw.numpy()
F_MIN, F_MAX = f_np[0], f_np[-1]

# 读取共振频率
pf = sample['geometry'].point_features[0]
n_modes = int(pf[5].item())
f_peaks = sorted([pf[6+k].item() * 600 for k in range(n_modes)])
pk1, pk2 = f_peaks[0], f_peaks[1]

# 非线性拉伸x轴：高斯权重
bw1, bw2 = 5.0, 8.0
w1 = 30.0 * np.exp(-0.5 * ((f_np - pk1) / (bw1 * 0.5))**2)
w2 =  6.0 * np.exp(-0.5 * ((f_np - pk2) / (bw2 * 0.5))**2)
tw = 1.0 + w1 + w2
xs = np.zeros_like(f_np)
for i in range(1, len(f_np)):
    xs[i] = xs[i-1] + (tw[i] + tw[i-1]) / 2 * (f_np[i] - f_np[i-1])

# 刻度
tick_freqs = np.unique(np.sort([F_MIN, pk1-bw1, pk1, pk1+bw1, pk2-bw2, pk2, pk2+bw2, F_MAX]))
tick_freqs = tick_freqs[(tick_freqs >= F_MIN) & (tick_freqs <= F_MAX)]
tick_locs = np.interp(tick_freqs, f_np, xs)
tick_labels = [f'{f:.1f}' for f in tick_freqs]

# 公共参数
x_positions = [0, 0.125, 0.25, 0.375, 0.5]
labels = ['固定端(x=0)', 'x=0.125', '中点(x=0.25)', 'x=0.375', '自由端(x=0.5)']
colors = plt.cm.viridis(np.linspace(0, 1, 5))
target_indices = [torch.argmin((points_3d[:,0] - x).abs()).item() for x in x_positions]

print(f'节点数: {points_3d.shape[0]}, 频率数: {len(freq_raw)}')
print(f'频率范围: [{F_MIN:.1f}, {F_MAX:.1f}] Hz')
print(f'峰值: {pk1:.1f} Hz, {pk2:.1f} Hz')

# ==========================================
# 图1: 幅值曲线 (拉伸轴)
# ==========================================
fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True)
fig.subplots_adjust(hspace=0.15)

for row, (ax, idx, label) in enumerate(zip(axes, target_indices, labels)):
    d = frf_amp[idx].numpy()
    ax.plot(xs, d, '.-', color=colors[row], linewidth=1.2, markersize=3)
    if x_positions[row] == 0:
        ax.set_ylim(-0.001, 0.001)
    else:
        ax.set_ylim(max(-50, d.min() * 1.15), d.max() * 1.15)
    ax.set_ylabel(label, fontsize=9)
    ax.grid(True, alpha=0.3)
    # 峰值区域遮罩
    s1 = [np.interp(pk1-bw1, f_np, xs), np.interp(pk1+bw1, f_np, xs)]
    s2 = [np.interp(pk2-bw2, f_np, xs), np.interp(pk2+bw2, f_np, xs)]
    ax.axvspan(s1[0], s1[1], color='gray', alpha=0.1)
    ax.axvspan(s2[0], s2[1], color='gray', alpha=0.1)

axes[-1].set_xticks(tick_locs)
axes[-1].set_xticklabels(tick_labels, fontsize=9)
axes[-1].set_xlabel('Frequency (Hz)', fontsize=10)
fig.suptitle('FRF Amplitude (Non-linear Stretched Axis)', fontsize=14, y=0.92)

out = os.path.join(out_dir, 'true_frf.png')
try: os.remove(out)
except OSError: pass
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f'图表1: {out}')

# ==========================================
# 图2: 峰值区域线性放大
# ==========================================
fig2, axes2 = plt.subplots(2, 1, figsize=(14, 10))
for ax, (lo, hi), pname in zip(axes2,
    [(pk1-bw1, pk1+bw1), (pk2-bw2, pk2+bw2)],
    [f'Peak 1 ({pk1:.1f} Hz)', f'Peak 2 ({pk2:.1f} Hz)']):
    fm = (freq_raw >= lo) & (freq_raw <= hi)
    for xp, c, label in zip(x_positions, colors, labels):
        idx = torch.argmin((points_3d[:,0] - xp).abs()).item()
        fz = freq_raw[fm].numpy()
        dz = frf_amp[idx][fm].numpy()
        ax.plot(fz, dz, 'o-', color=c, markersize=3, linewidth=1.2, label=label)
        pi = np.argmax(np.abs(dz))
        ax.annotate(f'{dz[pi]:.1f}', (fz[pi], dz[pi]),
                   textcoords='offset points', xytext=(0, 8), fontsize=7, color=c, ha='center')
    ax.set_xlabel('Frequency (Hz)'); ax.set_ylabel('FRF')
    ax.set_title(pname); ax.legend(fontsize=8, ncol=5); ax.grid(True, alpha=0.3)

fig2.tight_layout()
out2 = os.path.join(out_dir, 'peak_zoom.png')
try: os.remove(out2)
except OSError: pass
plt.savefig(out2, dpi=150, bbox_inches='tight')
plt.close()
print(f'图表2: {out2}')

# ==========================================
# 图3: Re/Im 平滑过零 (复数FRF核心优势)
# ==========================================
free_idx = torch.argmax(points_3d[:,0]).item()
re_d = frf_cmplx[free_idx, :, 0].numpy()
im_d = frf_cmplx[free_idx, :, 1].numpy()
am_d = frf_amp[free_idx].numpy()

fig3, (ar, ai, aa) = plt.subplots(3, 1, figsize=(14, 14), sharex=True)

ar.plot(xs, re_d, 'b-', linewidth=1.5)
ar.fill_between(xs, 0, re_d, alpha=0.15, color='blue')
ar.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ar.set_ylabel('Real Part'); ar.set_title(f'Real @ Free End (x={points_3d[free_idx,0]:.3f}m)')
ar.grid(True, alpha=0.3)

ai.plot(xs, im_d, 'r-', linewidth=1.5)
ai.fill_between(xs, 0, im_d, alpha=0.15, color='red')
ai.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ai.set_ylabel('Imaginary Part'); ai.set_title('Imaginary @ Free End')
ai.grid(True, alpha=0.3)

aa.plot(xs, am_d, 'k-', linewidth=1.5)
aa.set_ylabel('Amplitude'); aa.set_title('Amplitude = sqrt(Re^2 + Im^2)')
aa.grid(True, alpha=0.3)

aa.set_xticks(tick_locs)
aa.set_xticklabels(tick_labels, fontsize=9)
aa.set_xlabel('Frequency (Hz)')

fig3.tight_layout()
out3 = os.path.join(out_dir, 'complex_frf.png')
try: os.remove(out3)
except OSError: pass
plt.savefig(out3, dpi=150, bbox_inches='tight')
plt.close()
print(f'图表3: {out3}')
