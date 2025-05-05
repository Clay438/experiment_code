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

# 样条插值拟合
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

# 参数（全回流）
xD = 0.8847
xW = 0.069111
xF = 0.897555

# 绘制
x = np.linspace(0, 1, 500)
y = y_eq(x)

fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(x, y, label="Equilibrium Line", color="blue", lw=2)
ax.plot(x, x, label="45° Line", color="gray", linestyle="--", lw=1)
ax.plot(x, x, color="red", linestyle="-", label="Operating Line (Full Reflux)", lw=1.5)

# 标记点
ax.scatter([xD, xW, xF], [xD, xW, y_eq(xF)], 
           color=['purple', 'orange', 'green'], 
           marker='o', s=80, zorder=3,
           label=['Distillate (xD)', 'Bottoms (xW)', 'Feed (xF)'])

# 阶梯
stages = 0
x_prev = xD
y_prev = xD
stage_points = [(x_prev, y_prev)]

while True:
    x_curr = x_eq(y_prev)
    stage_points.append((x_curr, y_prev))
    stages += 1
    if x_curr <= xW + 1e-6:
        if x_curr < xW - 1e-3:
            print(f"Warning: Final x ({x_curr:.6f}) is below xW ({xW:.6f})")
        break
    y_curr = x_curr  # 全回流操作线
    stage_points.append((x_curr, y_curr))
    x_prev, y_prev = x_curr, y_curr

for i in range(0, len(stage_points)-1, 2):
    x_start, y_start = stage_points[i]
    x_end, y_end = stage_points[i+1]
    ax.hlines(y_start, x_end, x_start, colors='black', lw=1)
    if i+2 < len(stage_points):
        next_x, next_y = stage_points[i+2]
        ax.vlines(x_end, y_end, next_y, colors='black', lw=1)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel("Liquid mole fraction (x)")
ax.set_ylabel("Vapor mole fraction (y)")
ax.set_title(f"McCabe-Thiele Diagram (Full Reflux)\nTheoretical Stages: {stages-1} (including reboiler)")
ax.legend(loc='upper left')
ax.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()