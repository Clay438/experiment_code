import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 数据
Re = np.array([19863, 17643, 15543, 13357, 11149, 8989, 6765])
Nu = np.array([57.59, 52.73, 47.77, 42.31, 36.71, 30.99, 24.71])
Pr = np.array([0.728, 0.731, 0.733, 0.733, 0.735, 0.737, 0.738])
Nu_Pr = Nu / (Pr ** 0.4)

# 拟合函数
def func(Re, A, m):
    return A * Re**m

# 拟合
params, _ = curve_fit(func, Re, Nu_Pr)
A, m = params

# 拟合曲线
Re_fit = np.linspace(min(Re), max(Re), 200)
Nu_Pr_fit = func(Re_fit, A, m)

# 绘图
plt.figure(figsize=(8, 6))
plt.scatter(Re, Nu_Pr, color='blue', label='Experimental Data')
plt.plot(Re_fit, Nu_Pr_fit, color='red', label=f'Fit: Nu/Pr^0.4 = {A:.3e}·Re^{m:.3f}')
plt.xlabel('Reynolds number (Re)')
plt.ylabel('Nu / Pr^0.4')
plt.title('Regression of Nu / Pr^0.4 vs Re')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 输出结果
print(f"拟合常数 A = {A:.3e}")
print(f"拟合指数 m = {m:.3f}")
