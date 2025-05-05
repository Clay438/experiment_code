import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 实验数据
c = np.array([0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.16, 0.20, 0.24])  # 浓度 (mol/L)
Gamma = np.array([1.4704e-6, 2.4612e-6, 3.1743e-6, 3.7120e-6, 4.1319e-6,
                  4.4690e-6, 4.9764e-6, 5.3403e-6, 5.6139e-6])  # 表面吸附量 (mol/m²)

# Langmuir 吸附模型函数
def langmuir(c, Gamma_max, K):
    return (Gamma_max * K * c) / (1 + K * c)

# 提供初始猜测值
initial_guess = [max(Gamma), 100]

# 拟合数据
params, _ = curve_fit(langmuir, c, Gamma, p0=initial_guess, maxfev=5000)
Gamma_max_fit, K_fit = params

# 生成拟合曲线数据
c_fit = np.linspace(0, 0.25, 200)
Gamma_fit = langmuir(c_fit, Gamma_max_fit, K_fit)

# 绘图
plt.figure(figsize=(8, 6))
plt.scatter(c, Gamma, color='blue', label='data')
plt.plot(c_fit, Gamma_fit, color='red',
         label=f'fit line')
plt.xlabel('c (mol/L)')
plt.ylabel('Γ (mol/m²)')
plt.title('Γ ~ c fit line ')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
