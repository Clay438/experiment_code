import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial

# 原始数据
x_data = np.array([
    0.0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.14, 0.18, 0.20,
    0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70,
    0.75, 0.80, 0.85, 0.894, 0.90, 0.95, 1.00
])
y_data = np.array([
    0.0, 0.11, 0.17, 0.27, 0.34, 0.392, 0.43, 0.482, 0.513, 0.525,
    0.551, 0.575, 0.595, 0.614, 0.635, 0.657, 0.678, 0.698, 0.725, 0.755,
    0.785, 0.82, 0.855, 0.894, 0.898, 0.942, 1.00
])

# 多项式拟合（比如3次）
degree = 3
coeffs = np.polyfit(x_data, y_data, degree)

# 拟合曲线
poly = np.poly1d(coeffs)
x_fit = np.linspace(0, 1, 200)
y_fit = poly(x_fit)

# 绘图
plt.scatter(x_data, y_data, label='Data Points')
plt.plot(x_fit, y_fit, 'r-', label=f'{degree}-Degree Polynomial Fit')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

# 显示拟合方程
print(f"拟合方程：\n{poly}")
