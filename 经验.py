import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import linregress

# Raw data
Re = np.array([19863, 17643, 15543, 13357, 11149, 8989, 6765])
Nu = np.array([57.59, 52.73, 47.77, 42.31, 36.71, 30.99, 24.71])
Pr = np.array([0.728, 0.731, 0.733, 0.733, 0.735, 0.737, 0.738])
Nu_Pr = Nu / Pr**0.4

# Log transformation
ln_Re = np.log(Re)
ln_Nu_Pr = np.log(Nu_Pr)

# ===== Linear fit for experimental data =====
slope1, intercept1, r1, _, _ = linregress(ln_Re, ln_Nu_Pr)
y1_fit = slope1 * ln_Re + intercept1

# ===== Fit model Nu/Pr^0.4 = A * Re^m =====
def model_func(Re, A, m):
    return A * Re**m

params, _ = curve_fit(model_func, Re, Nu_Pr)
A, m = params
Nu_Pr_fitted = model_func(Re, A, m)
ln_Nu_Pr_fit = np.log(Nu_Pr_fitted)

# Linear fit for fitted model
slope2, intercept2, r2, _, _ = linregress(ln_Re, ln_Nu_Pr_fit)
y2_fit = slope2 * ln_Re + intercept2

# ===== Plotting =====
plt.figure(figsize=(10, 6))

# Experimental data points
plt.scatter(ln_Re, ln_Nu_Pr, color='blue', s=70, marker='o', label='Experimental Data Points')
# Fitted model data points
plt.scatter(ln_Re, ln_Nu_Pr_fit, color='green', s=70, marker='s', label='Model-Fitted Data Points')

# Linear fit lines
plt.plot(ln_Re, y1_fit, color='blue', linestyle='-', linewidth=2.5,
         label=f'Fit (Experimental): y = {slope1:.3f}·ln(Re) + {intercept1:.3f}')
plt.plot(ln_Re, y2_fit, color='green', linestyle='-', linewidth=0.5,
         label=f'Fit (Model): y = {slope2:.3f}·ln(Re) + {intercept2:.3f}')

# Labels and legend
plt.xlabel('ln(Re)', fontsize=12)
plt.ylabel('ln(Nu / Pr^0.4)', fontsize=12)
plt.title('Linear Regression: Experimental vs Model-Based', fontsize=14)
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()
plt.show()

# ===== Output fitting results =====
print(f"[Experimental Data Fit] ln(Nu/Pr^0.4) = {slope1:.4f} * ln(Re) + {intercept1:.4f}, R² = {r1**2:.4f}")
print(f"[Model-Based Fit]      ln(Nu/Pr^0.4) = {slope2:.4f} * ln(Re) + {intercept2:.4f}, R² = {r2**2:.4f}")
