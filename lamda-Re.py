import numpy as np
import matplotlib.pyplot as plt

Re_values = [18443,22162,25727,29137,32236,35646]
lambda_values = [0.1096,0.1172,0.1228,0.1286,0.1310,0.1326]


# 转换为 NumPy 数组
Re_values = np.array(Re_values)
lambda_values = np.array(lambda_values)

fit = np.polyfit(np.log10(Re_values), np.log10(lambda_values), 1)

# 计算拟合曲线
Re_fit = np.logspace(np.log10(Re_values.min()), np.log10(Re_values.max()), 100)
lambda_fit = 10**(fit[1]) * Re_fit**(fit[0])

# 绘图
plt.figure(figsize=(8, 6))

# 画实验数据点
plt.loglog(Re_values, lambda_values, 'bo', label="data")

# 画拟合曲线
plt.loglog(Re_fit, lambda_fit, 'b-', label=f": λ = {10**fit[1]:.4f} * Re^{fit[0]:.2f}")

# 设置轴标签和标题
plt.xlabel(r' $Re$')
plt.ylabel(r' $\lambda$')
plt.title('λ~Re')

# 显示网格和图例
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.legend()
plt.show()