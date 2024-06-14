import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 손실 함수 정의 (안장점을 가진 2차원 손실 함수 예시)
def loss_function(x, y):
    return x**2 - y**2

# 그리드 생성
x_range = np.linspace(-5, 5, 100)
y_range = np.linspace(-5, 5, 100)
x, y = np.meshgrid(x_range, y_range)
z = loss_function(x, y)

# 3D 플롯 생성
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# 손실 함수의 3D 플롯 생성
ax.plot_surface(x, y, z, cmap='viridis', zorder=1)

# # 안장점 표시 (여기서는 (0, 0)이 안장점으로 가정)
# ax.scatter([0], [0], [loss_function(0, 0)], color='red', marker='o', s=30, label='Saddle Point', zorder=2)

# 그래프 설정
ax.set_xlabel(r'$\theta_1$')
ax.set_ylabel(r'$\theta_2$')
# ax.set_zlabel('Loss')
ax.set_title('Saddle point')
ax.text2D(0.02, 0.72, 'Loss', transform=ax.transAxes)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])

# 플롯 보이기
plt.savefig('/root/PyHessian/graph/saddle_point.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
