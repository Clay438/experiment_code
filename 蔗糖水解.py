import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# 原始数据（时间段、中间时间 t、Δα、|Δα|、ln|Δα|）
data = {
    "time_range": ["5→10", "10→15", "15→20", "20→25", "25→30", "30→35", "35→40"],
    "t": [7.5, 12.5, 17.5, 22.5, 27.5, 32.5, 37.5],
    "delta_alpha": [-5.90, -2.40, -0.57, -2.65, -1.11, -0.01, -0.28],
    "abs_delta_alpha": [5.90, 2.40, 0.57, 2.65, 1.11, 0.01, 0.28],
    "ln_abs_delta_alpha": [1.7749, 0.8755, -0.5616, 0.9734, 0.1044, -4.6052, -1.2730]
}

# 转换为NumPy数组
t = np.array(data["t"])
ln_abs = np.array(data["ln_abs_delta_alpha"])

# 剔除异常点（30→35 min，索引5）
mask = np.ones(len(t), dtype=bool)
mask[5] = False  # 剔除第6个数据点（Python索引从0开始）
t_clean = t[mask]
ln_abs_clean = ln_abs[mask]

# 线性拟合：ln|Δα| = k * t + C
slope, intercept, r_value, p_value, std_err = stats.linregress(t_clean, ln_abs_clean)

# 拟合结果
print("拟合结果:")
print(f"斜率 k = {slope:.4f}")
print(f"截距 C = {intercept:.4f}")
print(f"相关系数 R² = {r_value**2:.4f}")
print(f"标准误差 = {std_err:.4f}")

# 绘制原始数据和拟合直线
plt.figure(figsize=(8, 5))
plt.scatter(t_clean, ln_abs_clean, color='red', label='data')
plt.plot(t_clean, slope * t_clean + intercept, 'b-', label=f'linear fit: ln|Δα| = {slope:.3f}t + {intercept:.3f}')
plt.xlabel('t (min)')
plt.ylabel('ln|Δα|')
plt.title('ln|Δα| ~ t ')
plt.legend()
plt.grid(True)
plt.show()

# 输出拟合的预测值
print("\n预测值对比:")
for i in range(len(t_clean)):
    predicted = slope * t_clean[i] + intercept
    print(f"t = {t_clean[i]:.1f} min: 实际值 = {ln_abs_clean[i]:.4f}, 拟合值 = {predicted:.4f}")