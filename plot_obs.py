import numpy as np
import matplotlib.pyplot as plt

# # 假设 data 是 n 行 39 列的 NumPy 数组
# data = np.loadtxt("obs_buf.txt", delimiter=",")

# # 按照给定的列划分
# base_quat = data[:, 0:4]  # 4列
# base_ang_vel = data[:, 4:7]  # 3列
# commands = data[:, 7:11]  # 4列
# theta0 = data[:, 11:13]  # 2列
# theta0_dot = data[:, 13:15]  # 2列
# L0 = data[:, 15:17]  # 2列
# L0_dot = data[:, 17:19]  # 2列
# dof_pos = data[:, 19:23]  # 4列
# dof_vel = data[:, 23:31]  # 8列
# actions = data[:, 31:39]  # 8列


# # 创建一个图形
# plt.figure(figsize=(10, 6))

# # 为 base_quat 的每列绘制一条线
# for i in range(2):
#     plt.plot((L0[:, i]), label=f"L0 {i+1}")

# # 设置标题和图例
# plt.title("Data Over Time")
# plt.xlabel("Time Step")
# plt.ylabel("Value")
# plt.legend()
# plt.grid(True)

# # 显示图形
# plt.tight_layout()
# plt.show()


# 假设 data 是 n 行 39 列的 NumPy 数组
data = np.loadtxt("privileged_obs_buf.txt", delimiter=",")

# 按照给定的列划分
base_lin_vel = data[:, 0:3]
obs_buf = data[:, 3:42]
projected_gravity = data[:, 42:45]
last_action = data[:, 45:53]
last_last_action = data[:, 53:61]
dof_acc = data[:, 61:69]
heights = data[:, 69:146]
torques = data[:, 146:154]
mass_added = data[:, 154]
base_com = data[:, 155:158]
friction_coef = data[:, 158]
restitution_coef = data[:, 159]
ext_ft = data[:, 160:166]

base_quat = obs_buf[:, 0:4]  # 4列
base_ang_vel = obs_buf[:, 4:7]  # 3列
commands = obs_buf[:, 7:11]  # 4列
theta0 = obs_buf[:, 11:13]  # 2列
theta0_dot = obs_buf[:, 13:15]  # 2列
L0 = obs_buf[:, 15:17]  # 2列
L0_dot = obs_buf[:, 17:19]  # 2列
dof_pos = obs_buf[:, 19:23]  # 4列
dof_vel = obs_buf[:, 23:31]  # 8列
actions = obs_buf[:, 31:39]  # 8列


# 创建一个图形
plt.figure(figsize=(10, 6))


for i in [3, 7]:
    plt.plot((actions[:, i]), label=f"actions {i+1}")

# 设置标题和图例
plt.title("Data Over Time")
plt.xlabel("Time Step")
plt.ylabel("Value")
plt.legend()
plt.grid(True)

# 显示图形
plt.tight_layout()
plt.show()

# # 创建一个图形
# plt.figure(figsize=(10, 6))

# # 为 base_quat 的每列绘制直方图
# for i in range(1):
#     plt.hist((mass_added[:, i] / 1) * 1, bins=30, alpha=0.7, label=f"mass_added {i+1}")

# # 设置标题和图例
# plt.title("Data Histogram")
# plt.xlabel("Value")
# plt.ylabel("Frequency")
# plt.legend()
# plt.grid(True)

# # 显示图形
# plt.tight_layout()
# plt.show()
