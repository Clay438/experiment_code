import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.interpolate import UnivariateSpline

# 原始气液平衡数据
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

# 样条拟合平衡曲线
spline = UnivariateSpline(x_data, y_data, k=3, s=0.005)
def y_eq(x):
    return np.clip(spline(x), 0, 1)

def x_eq(y_target):
    solutions = []
    for y in np.atleast_1d(y_target):
        if y >= 1.0:
            solutions.append(1.0)
        elif y <= 0.0:
            solutions.append(0.0)
        else:
            x_guess = np.interp(y, y_data, x_data)
            sol, *_ = fsolve(lambda x: y_eq(x) - y, x0=x_guess, full_output=True)
            solutions.append(np.clip(sol[0], 0, 1))
    return np.array(solutions).item() if np.isscalar(y_target) else np.array(solutions)


xF = 0.102
xD = 0.6657
xW = 0.0056
R = 2  # 回流比
q = 1.148  # 进料热状况参数

# A步：画出气液平衡曲线+对角线
x = np.linspace(0, 1, 500)
y = y_eq(x)

fig, ax = plt.subplots(figsize=(8,8))
ax.plot(x, y, label="Equilibrium Curve", color="blue", lw=2)
ax.plot(x, x, label="45° Line", color="gray", linestyle="--", lw=1)

# B步：在x轴上标出xD、xF、xW并作垂线交对角线
def point_on_diag(x):
    return (x, x)

a = point_on_diag(xD)
f = point_on_diag(xF)
b = point_on_diag(xW)

ax.scatter(*a, color='black', label='Point a')
ax.scatter(*f, color='black', label='Point f')
ax.scatter(*b, color='black', label='Point b')

# C步：画精馏段操作线
yC = xD / (R + 1)  # 在y轴上yC位置
c = (0, yC)

def rectifying_line(x):
    return (R/(R+1))*x + xD/(R+1)

ax.plot([c[0], a[0]], [c[1], a[1]], color="red", label="Rectifying Operating Line", lw=1.5)

# D步：画q线，求交点d
# D步：画q线，求交点d
# D步：画q线，只从f点向右延伸
q_slope = q / (q - 1)
def q_line(x):
    return q_slope * (x - xF) + xF

# 求q线和精馏段操作线的交点d
def intersection(x):
    return rectifying_line(x) - q_line(x)

x_d = fsolve(intersection, xF)[0]
y_d = rectifying_line(x_d)
d = (x_d, y_d)
ax.scatter(*d, color='blue', label='Point d')

# 正确画q线（只从f向右延伸到1）
x_q = np.linspace(xF, 1, 100)
ax.plot(x_q, q_line(x_q), color="green", lw=1.5, label="q-line")



# E步：连接d和b作提馏段操作线
def stripping_line(x):
    m = (b[1] - d[1]) / (b[0] - d[0])
    return m*(x - b[0]) + b[1]

ax.plot([d[0], b[0]], [d[1], b[1]], color="purple", label="Stripping Operating Line", lw=1.5)

# F步：从a开始画阶梯
stages = 0
stages_rectifying = 0  # 精馏段板数
stages_stripping = 0   # 提馏段板数

x_curr, y_curr = a
ax.scatter(x_curr, y_curr, color='red', s=50)  # 起点标注
while x_curr > xW:
    # 水平到平衡曲线
    x_new = x_eq(y_curr)
    ax.hlines(y_curr, x_new, x_curr, colors='black')
    x_curr = x_new

    # 判断是在精馏段还是提馏段
    if x_curr >= d[0]:  # 精馏段
        y_new = rectifying_line(x_curr)
        stages_rectifying += 1
    else:               # 提馏段
        y_new = stripping_line(x_curr)
        stages_stripping += 1

    ax.vlines(x_curr, y_curr, y_new, colors='black')
    y_curr = y_new
    stages += 1

# G步：最终统计
stages += 1  # 包括再沸器
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel("Liquid Mole Fraction x", fontsize=12)
ax.set_ylabel("Vapor Mole Fraction y", fontsize=12)
ax.set_title(f"McCabe-Thiele Diagram\nTotal Theoretical Stages: {stages} (including reboiler)\nRectifying stages: {stages_rectifying}, Stripping stages: {stages_stripping}", fontsize=14)
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend()
plt.tight_layout()
plt.show()
