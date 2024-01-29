import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler

from rl_mus.agents.uav import UavCtrlType


class Plotter:
    def __init__(self, num_uavs=1, ctrl_type=UavCtrlType.VEL, freq=240) -> None:
        self.ctrl_type = ctrl_type
        self.num_uavs = num_uavs

        # used for converting the uav_id to array index
        self.uav_ids = {}
        self.data = [{"ctrl": [], "state": []} for i in range(self.num_uavs)]
        self.num_time_steps = 0
        self.freq = freq
        self.uav_counter = 0

    def add_uav(self, uav_id):
        self.uav_ids[uav_id] = self.uav_counter
        self.uav_counter += 1

    def log(self, uav_id, state, ref_ctrl):
        array_idx = self.uav_ids[uav_id]
        self.data[array_idx]["state"].append(state)
        self.data[array_idx]["ctrl"].append(ref_ctrl)

    def plot(self, title="", plt_ctrl=False):
        if self.num_uavs > 1:
            plt.rc(
                "axes",
                prop_cycle=(
                    cycler("color", ["r", "g", "b", "y"])
                    + cycler("linestyle", ["-", "--", ":", "-."])
                ),
            )

        num_rows = 8
        num_cols = 2

        self.fig, self.axs = plt.subplots(
            num_rows, num_cols, sharex=True, figsize=(14, 12)
        )
        self.fig.suptitle(title)

        # convert data to numpy arrays
        for uav_id in range(self.num_uavs):
            self.data[uav_id]["state"] = np.array(self.data[uav_id]["state"])
            self.data[uav_id]["ctrl"] = np.array(self.data[uav_id]["ctrl"])

        self.num_time_steps = self.data[0]["state"].shape[0]

        col = 0
        # x, y, z
        row = 0
        self.plot_uav_data(row, col, 0, ylabel="x (m)")
        row = 1
        self.plot_uav_data(row, col, 1, ylabel="y (m)")
        row = 2
        self.plot_uav_data(row, col, 2, ylabel="z (m)")

        # roll, pitch, yaw
        row = 3
        self.plot_uav_data(row, col, 7, ylabel="$\phi$ (rad)")
        row = 4
        self.plot_uav_data(row, col, 8, ylabel=r"$\theta$ (rad)")
        row = 5
        self.plot_uav_data(row, col, 9, ylabel="$\psi$ (rad)")

        # vel and vel commands
        col = 1
        row = 0
        self.plot_uav_data(row, col, 10, ylabel="vx (m/s)")
        row = 1
        self.plot_uav_data(row, col, 11, ylabel="vy (m/s)")
        row = 2
        self.plot_uav_data(row, col, 12, ylabel="vz (m/s)")

        # angular velocitys
        col = 1
        row = 3
        self.plot_uav_data(row, col, 13, ylabel="r (rad/s)")
        row = 4
        self.plot_uav_data(row, col, 14, ylabel="p (rad/s)")
        row = 5
        self.plot_uav_data(row, col, 15, ylabel="q (rad/s)")

        # RPMS
        col = 0
        row = 6
        self.plot_uav_data(row, col, 16, ylabel="RPM0")
        row = 7
        self.plot_uav_data(row, col, 17, ylabel="RPM1")

        col = 1
        row = 6
        self.plot_uav_data(row, col, 18, ylabel="RPM2")
        row = 7
        self.plot_uav_data(row, col, 19, ylabel="RPM3")

        if plt_ctrl:
            if self.ctrl_type == UavCtrlType.VEL:
                col = 1
                row = 0
                self.plot_uav_data(row, col, 0, data_type="ctrl", ylabel="vx (m/s)")
                row = 1
                self.plot_uav_data(row, col, 1, data_type="ctrl", ylabel="vy (m/s)")
                row = 2
                self.plot_uav_data(row, col, 2, data_type="ctrl", ylabel="vz (m/s)")

            elif self.ctrl_type == UavCtrlType.POS:
                col = 0
                row = 0
                self.plot_uav_data(row, col, 0, data_type="ctrl", ylabel="x (m)")
                row = 1
                self.plot_uav_data(row, col, 1, data_type="ctrl", ylabel="y (m)")
                row = 2
                self.plot_uav_data(row, col, 2, data_type="ctrl", ylabel="z (m)")
                row = 5
                self.plot_uav_data(row, col, 3, data_type="ctrl", ylabel="$\psi$ (rad)")

        for row in range(num_rows):
            for col in range(num_cols):
                self.axs[row, col].grid(True)
                self.axs[row, col].legend(loc="upper right", frameon=True)

        self.fig.subplots_adjust(hspace=0)

        plt.show()

    def plot_uav_data(
        self,
        row,
        col,
        data_idx,
        data_type="state",
        ylabel="",
    ):
        t = np.arange(self.num_time_steps) / self.freq
        for i in range(self.num_uavs):
            self.axs[row, col].plot(
                t, self.data[i][data_type][:, data_idx], label=f"uav_{i}"
            )
        self.axs[row, col].set_xlabel("t (s)")
        self.axs[row, col].set_ylabel(ylabel)


def plot_traj(uav_des_traj, uav_trajectory, title="", scale=240.0):
    fig, axs = plt.subplots(4, 2, sharex=False, figsize=(12, 10), layout="constrained")
    t_axis = np.arange(uav_trajectory.shape[0]) / scale

    axs[0, 0].plot(t_axis, uav_des_traj[:, 0])
    axs[0, 0].plot(t_axis, uav_trajectory[:, 0])
    axs[0, 0].set_xlabel("t (s)")
    axs[0, 0].set_ylabel("x (m)")

    axs[0, 1].plot(t_axis, uav_des_traj[:, 10])
    axs[0, 1].plot(t_axis, uav_trajectory[:, 10])
    axs[0, 1].set_xlabel("t(s)")
    axs[0, 1].set_ylabel("$\dot{x}$ (m/s)")

    axs[1, 0].plot(t_axis, uav_des_traj[:, 1])
    axs[1, 0].plot(t_axis, uav_trajectory[:, 1])
    axs[1, 0].set_xlabel("t(s)")
    axs[1, 0].set_ylabel("y (m)")

    axs[1, 1].plot(t_axis, uav_des_traj[:, 11])
    axs[1, 1].plot(t_axis, uav_trajectory[:, 11])
    axs[1, 1].set_xlabel("t(s)")
    axs[1, 1].set_ylabel("$\dot{y}$ (m/s)")

    axs[2, 0].plot(t_axis, uav_des_traj[:, 2])
    axs[2, 0].plot(t_axis, uav_trajectory[:, 2])
    axs[2, 0].set_xlabel("t(s)")
    axs[2, 0].set_ylabel("z (m)")

    axs[2, 1].plot(t_axis, uav_des_traj[:, 12])
    axs[2, 1].plot(t_axis, uav_trajectory[:, 12])
    axs[2, 1].set_xlabel("t(s)")
    axs[2, 1].set_ylabel("$\dot{z}$ (m/s)")

    axs[3, 0].plot(t_axis, uav_des_traj[:, 9])
    axs[3, 0].plot(t_axis, uav_trajectory[:, 9])
    axs[3, 0].set_xlabel("t(s)")
    axs[3, 0].set_ylabel("$\psi$ (rad)")

    axs[3, 1].plot(t_axis, uav_des_traj[:, 15])
    axs[3, 1].plot(t_axis, uav_trajectory[:, 15])
    axs[3, 1].set_xlabel("t(s)")
    axs[3, 1].set_ylabel("$\dot{\psi}$ (rad/s)")

    fig.suptitle(title, fontsize=16)

    plt.show()
