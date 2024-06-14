import numpy as np
import matplotlib.pyplot as plt

# Flat한 Minima를 나타내는 예시 손실 함수
def flat_minima_loss(x, y):
    return 2*x**2 + 2*y**2

# Sharp한 Minima를 나타내는 예시 손실 함수
def sharp_minima_loss(x, y):
    return 4*x**2 + 4*y**2

# 그리드 생성
x_range = np.linspace(-2, 2, 100)
y_range = np.linspace(-2, 2, 100)
x, y = np.meshgrid(x_range, y_range)
z_flat_minima = flat_minima_loss(x, y)
z_sharp_minima = sharp_minima_loss(x, y)

# 3D 플롯 생성
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Flat한 Minima를 나타내는 그래프
ax.plot_surface(x, y, z_flat_minima, cmap='viridis')
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(0, 30)
ax.set_xlabel(r'$w_1$')
ax.set_ylabel(r'$w_2$')
ax.set_zlabel('Loss')
ax.text2D(0.02, 0.72, 'Loss', transform=ax.transAxes)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])
ax.set_title('Flat minima landscape')

plt.savefig('/root/PyHessian/graph/flat_sharp_compare_flat.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.clf()

# 3D 플롯 생성
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Sharp한 Minima를 나타내는 그래프
ax.plot_surface(x, y, z_sharp_minima, cmap='viridis')

# Flat Minima Landscape와 동일한 범위로 x축과 y축 설정
ax.set_zticks(np.arange(0, 31, 5))
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(0, 30)
ax.set_xlabel(r'$w_1$')
ax.set_ylabel(r'$w_2$')
ax.set_zlabel('Loss')
ax.text2D(0.02, 0.72, 'Loss', transform=ax.transAxes)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])
ax.set_title('Sharp Minima Landscape')

# # 그래프 설정
# for ax in axes:
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Loss')

# 그래프 저장
plt.savefig('/root/PyHessian/graph/flat_sharp_compare_sharp.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
