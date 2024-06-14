import numpy as np
import matplotlib.pyplot as plt

# 볼록한 함수 예시: y = x^2
def convex_function(x):
    return x**2

def non_convex_function_with_higher_peaks(x):
    return (x-1)*(x+2)*(x+3)*(x-4) + 4

# 그래프 그리기
x_values = np.linspace(-5, 5, 100)

# 볼록한 함수의 그래프
convex_values = convex_function(x_values)
plt.plot(x_values, convex_values, color='orange', label='Convex Function')
plt.scatter(0, convex_function(0), color='red', marker='o', label='Global Minima', zorder=2)
plt.text(0.1, convex_function(0)+0.5, r'$w^*$', color='black', ha='center', va='bottom')
plt.title('Convex Optimization Problem')
plt.xlabel('$\it{w}$')
plt.ylabel('Loss')
plt.xticks([])
plt.yticks([])
plt.legend()

plt.tight_layout()
plt.savefig('/root/PyHessian/graph/convex_graph.png') 
plt.clf()

# 볼록하지 않은 함수의 그래프
non_convex_values = non_convex_function_with_higher_peaks(x_values)
plt.plot(x_values, non_convex_values, label='Non-Convex Function', color='orange', zorder=1)
plt.scatter(2.89, non_convex_function_with_higher_peaks(2.89), color='red', marker='o', label='Global Minima', zorder=2)
plt.scatter(-2.55, non_convex_function_with_higher_peaks(-2.55), color='blue', marker='o', label='Local Minima', zorder=2)
plt.legend()
plt.title('Non-Convex Optimization Problem')
plt.xlabel(r'$w$')
plt.ylabel('Loss')
plt.xticks([])
plt.yticks([])
plt.legend()

plt.tight_layout()
plt.savefig('/root/PyHessian/graph/non_convex_graph.png') 
