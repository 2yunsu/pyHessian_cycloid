import numpy as np
import matplotlib.pyplot as plt
import torch

num_epochs = 100

x = torch.linspace(-2, 2, 100).view(-1,1)
mu = 0.02

theta_1 = torch.linspace(1 * np.pi, 2 * np.pi, num_epochs).view(-1,1)
theta_2 = torch.linspace(0, 1 * np.pi, num_epochs).view(-1,1)

r = 1  # Cycloid 그래프의 높이 맞추기
x_1 = r * (theta_1 - np.sin(theta_1))
x_2 = r * (theta_2 - np.sin(theta_2))
cycloid_graph_1 = -r * (1 - np.cos(theta_1))
cycloid_graph_2 = -r * (1 - np.cos(theta_2))

#위치 조정
x_1 = (x_1 / x_2[-1])
x_2 = (x_2 / x_2[-1])

fric_x_1 = r * (theta_1 - np.sin(theta_1)) + (mu * r * (1 - np.cos(theta_1)))
fric_x_2 = r * (theta_2 - np.sin(theta_2)) + (mu * r * (1 - np.cos(theta_2)))
fric_cycloid_graph_1 = -r * (1 - np.cos(theta_1)) + (mu * r * (theta_1 + np.sin(theta_1)))
fric_cycloid_graph_2 = -r * (1 - np.cos(theta_2)) + (mu * r * (theta_2 + np.sin(theta_2)))

#위치 조정
fric_x_1 = (fric_x_1 / fric_x_2[-1]) * (x_1[-1] - x_1[0]) + x_2[0]
fric_x_2 = (fric_x_2 / fric_x_2[-1]) * (x_2[-1] - x_2[0]) + x_2[0]
fric_cycloid_graph_1_1 = fric_cycloid_graph_1 + cycloid_graph_1[0] #y축 왼쪽 시작점 맞추기
fric_cycloid_graph_2_1 = fric_cycloid_graph_2 + cycloid_graph_2[0]


#compare gradient graph
plt.plot(x_1, cycloid_graph_1, label="Cycloid", color="C1")
plt.plot(x_2, cycloid_graph_2, color="C1")
plt.plot(fric_x_1, fric_cycloid_graph_1, label="Frictional Cycloid", color="C0")
plt.plot(fric_x_2, fric_cycloid_graph_2, color="C0")
plt.title(r"Cycloid & Frictional Cycloid, $\mu$ = {}".format(mu))
plt.xlabel(r"$w$")
plt.ylabel("Loss")
plt.legend()
plt.savefig('/root/PyHessian/graph/cycloid_and_frictional_cycloid_graph.png', dpi=300, bbox_inches='tight', pad_inches=0.1)