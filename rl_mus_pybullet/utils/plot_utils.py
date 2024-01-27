import matplotlib.pyplot as plt
import numpy as np

def plot_traj(uav_des_traj, uav_trajectory, title="", scale=240.):
    fig, axs = plt.subplots(4,2, sharex=False, figsize=(12,10), layout='constrained')
    t_axis = np.arange(uav_trajectory.shape[0]) / scale

    axs[0,0].plot(t_axis, uav_des_traj[:, 0])
    axs[0,0].plot(t_axis, uav_trajectory[:, 0])
    axs[0,0].set_xlabel("t(s)")
    axs[0,0].set_ylabel("x (m)")

    axs[0,1].plot(t_axis, uav_des_traj[:, 10])
    axs[0,1].plot(t_axis, uav_trajectory[:, 10])
    axs[0,1].set_xlabel("t(s)")
    axs[0,1].set_ylabel("$\dot{x}$ (m/s)")

    axs[1,0].plot(t_axis, uav_des_traj[:, 1])
    axs[1,0].plot(t_axis, uav_trajectory[:, 1])
    axs[1,0].set_xlabel("t(s)")
    axs[1,0].set_ylabel("y (m)")

    axs[1,1].plot(t_axis, uav_des_traj[:, 11])
    axs[1,1].plot(t_axis, uav_trajectory[:, 11])
    axs[1,1].set_xlabel("t(s)")
    axs[1,1].set_ylabel("$\dot{y}$ (m/s)")

    axs[2,0].plot(t_axis, uav_des_traj[:, 2])
    axs[2,0].plot(t_axis, uav_trajectory[:, 2])
    axs[2,0].set_xlabel("t(s)")
    axs[2,0].set_ylabel("z (m)")

    axs[2,1].plot(t_axis, uav_des_traj[:, 12])
    axs[2,1].plot(t_axis, uav_trajectory[:, 12])
    axs[2,1].set_xlabel("t(s)")
    axs[2,1].set_ylabel("$\dot{z}$ (m/s)")

    axs[3,0].plot(t_axis, uav_des_traj[:, 9])
    axs[3,0].plot(t_axis, uav_trajectory[:, 9])
    axs[3,0].set_xlabel("t(s)")
    axs[3,0].set_ylabel("$\psi$ (rad)")

    axs[3,1].plot(t_axis, uav_des_traj[:, 15])
    axs[3,1].plot(t_axis, uav_trajectory[:, 15])
    axs[3,1].set_xlabel("t(s)")
    axs[3,1].set_ylabel("$\dot{\psi}$ (rad/s)")

    fig.suptitle(title, fontsize=16)

    plt.show()