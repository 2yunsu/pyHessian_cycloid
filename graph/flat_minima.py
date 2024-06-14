import numpy as np
import matplotlib.pyplot as plt

# 손실 함수 정의
def loss_function(x):
    return (x-2.3)*(x+2)*x*x
# 그리드 생성
x_range = np.linspace(-4, 4, 100)
y = loss_function(x_range)

# 플롯 생성
plt.figure(figsize=(8, 6))
plt.plot(x_range, y, label='Loss Function')

# 그래프 설정
plt.xlabel('w')
plt.ylabel('Loss')
plt.title('Loss landscape with Flat Minima')

# 플롯 보이기
plt.savefig('/root/PyHessian/graph/flat_minima.png')
