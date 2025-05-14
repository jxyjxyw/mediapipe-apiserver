import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfilt, sosfilt_zi

class LowPassFilter:
    def __init__(self, cutoff=5, fs=180, order=4, input_shape=(3,)):
        """
        :param cutoff: 截止频率（Hz）
        :param fs: 采样频率（Hz）
        :param order: 滤波器阶数
        :param input_shape: 输入数组的形状
        """
        self.input_shape = input_shape
        self.flat_size = np.prod(input_shape)
        self.sos = butter(order, cutoff / (0.5 * fs), btype='low', output='sos')
        zi_single = sosfilt_zi(self.sos)  # shape: [sections, 2]
        self.zi = np.tile(zi_single[:, :, np.newaxis], (1, 1, self.flat_size))

    def filter(self, x):
        """
        :param x: 实时输入的形状为 input_shape 的 ndarray
        :return: 滤波后的结果，形状同输入
        """
        x = np.asarray(x)
        if x.shape != self.input_shape:
            raise ValueError(f"输入 shape 应为 {self.input_shape}，但收到 {x.shape}")

        x_flat = x.flatten()[np.newaxis, :]  # shape: [1, flat_size]
        y_flat = np.zeros_like(x_flat)

        for i in range(self.flat_size):
            y_flat[:, i], self.zi[:, :, i] = sosfilt(self.sos, x_flat[:, i], zi=self.zi[:, :, i])

        return y_flat.reshape(self.input_shape)


j3ds = np.load("j3d.npy")
j3ds_left = np.load("j3d_left.npy")
j3ds_right = np.load("j3d_right.npy")

# plot 33 subfigures in 3 colums
fig, axs = plt.subplots(6, 6, figsize=(30, 30))
# fig.subplots_adjust(hspace=0.5, wspace=0.5)

# filter = LowPassFilter(cutoff=5, fs=60, order=4, input_shape=(33, 3))
# j3ds_after = []
# for i in range(len(j3ds)):
#     j3ds_after.append((filter.filter(j3ds[i])))

# j3ds_after = np.array(j3ds_after)

for joint_idx in range(33):
    j3d = j3ds[:, joint_idx, :]
    ax = axs[joint_idx // 6, joint_idx % 6]
    ax.plot(j3d[:, 0], j3d[:, 2], c='r', label='fused')
    ax.plot(j3ds_left[:, joint_idx, 0], j3ds_left[:, joint_idx, 2], c='b', label='left')
    ax.plot(j3ds_right[:, joint_idx, 0], j3ds_right[:, joint_idx, 2], c='g', label='right')
    # ax.plot(j3ds_after[:, joint_idx, 0], j3ds_after[:, joint_idx, 2], c='r')
    ax.set_title(f"Joint {joint_idx}")
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_xlim(-3, 3)
    ax.set_ylim(1, 7)
    # ax.set_aspect('equal', adjustable='box')
    ax.grid()

mean = np.mean(j3ds[:, :13], axis=1)
mean_2 = np.mean(j3ds[:, (23,24)], axis=1)
ax = axs[5, 5]
ax.plot(mean[:, 0], mean[:, 2], c='r', label='fused')
ax.plot(mean_2[:, 0], mean_2[:, 2], c='b', label='left')
# ax.plot(mean[:, 0], mean[:, 2], c='b', label='left')
# ax.plot(mean[:, 0], mean[:, 2], c='g', label='right')

# plt.clf()
# fig, axs = plt.subplots(6, 6, figsize=(30, 30))
# # plt the y axis
# for joint_idx in range(33):
#     j3d = j3ds[:, joint_idx, :]
#     ax = axs[joint_idx // 6, joint_idx % 6]
#     ax.plot(j3d[:, 1], c='r', label='fused')
#     ax.plot(j3ds_left[:, joint_idx, 1], c='b', label='left')
#     ax.plot(j3ds_right[:, joint_idx, 1], c='g', label='right')
#     # ax.plot(j3ds_after[:, joint_idx, 0], j3ds_after[:, joint_idx, 1], c='r')
#     ax.set_title(f"Joint {joint_idx}")
#     ax.set_xlabel("_")
#     ax.set_ylabel("Y")
#     ax.set_xlim(0, 1000)
#     ax.set_ylim(-1, 1)
#     # ax.set_aspect('equal', adjustable='box')
#     ax.grid()


plt.tight_layout()
plt.show()