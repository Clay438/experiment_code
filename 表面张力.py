import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 原始数据
concentration = np.array([0,0.02,0.04,0.06,0.08,0.1,0.12,0.16,0.2,0.24])
gamma_values = np.array([71.60,69.73,65.77,62.03,60.25,56.93,55.05,52.24,49.43,47.77])

# 拟合函数定义
def gamma_func(c, k, a, b):
    return k - k * b * np.log(1 + c / a)

# 曲线拟合
initial_guess = [73.0, 0.01, 0.2]
params, covariance = curve_fit(gamma_func, concentration, gamma_values, p0=initial_guess)
k_fit, a_fit, b_fit = params

# 拟合曲线
c_fit = np.linspace(0, 0.25, 300)
gamma_fit = gamma_func(c_fit, *params)

# 绘图
plt.figure(figsize=(8, 5))
plt.scatter(concentration, gamma_values, color='blue', label='Data', s=60)
plt.plot(c_fit, gamma_fit, color='red', linestyle='--', label='Fit')

# 公式文本（直接带入数值）
fit_formula = (r'$\gamma(c) = {:.5f} - {:.5f} \cdot {:.5f} \cdot \ln\left(1 + \frac{{c}}{{{:.5f}}}\right)$'
               .format(k_fit, k_fit, b_fit, a_fit))

# 显示公式在拟合曲线附近（坐标可调）
plt.text(0.03, gamma_func(0.03, *params) + 2, fit_formula, fontsize=12,
         bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.4'))

# 图形美化
plt.xlabel('c (mol/L)', fontsize=12)
plt.ylabel('Surface Tension (mN/m)', fontsize=12)
plt.title('Nonlinear Fitting of Surface Tension vs. Concentration', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()
print(f"拟合结果: k = {k_fit:.3f}, a = {a_fit:.5f}, b = {b_fit:.3f}")