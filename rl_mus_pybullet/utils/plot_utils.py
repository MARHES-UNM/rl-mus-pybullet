import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler

from rl_mus_pybullet.agents.uav import UavCtrlType


class Plotter:
    def __init__(self, num_uavs=1, ctrl_type=UavCtrlType.VEL, freq=240) -> None:
        self.num_uavs = num_uavs
        self.uav_ref_ctrl = []
        self.uav_state = []
        # used for converting the uav_id to array index
        self.uav_ids = {}
        self.data = [{"ctrl": [], "state": []} for i in range(self.num_uavs)]
        self.num_time_steps = 0
        self.freq = freq

    def log(self, uav_id, state, ref_ctrl=None):
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
        # self.fig, self.axs = plt.subplots(num_rows, num_cols, sharex=True, figsize=(12, 10), layout="constrained")
        self.fig, self.axs = plt.subplots(
            num_rows, num_cols, sharex=True, figsize=(12, 10)
        )

        # convert data to numpy arrays
        for uav_id in range(self.num_uavs):
            self.data[uav_id]["state"] = np.array(self.data[uav_id]["state"])
            self.data[uav_id]["ctrl"] = np.array(self.data[uav_id]["ctrl"])
        col = 0

        self.num_time_steps = self.data[0]["state"].shape[0]

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

        if plt_ctrl:
            row = 0
            self.plot_uav_data(row, col, 0, data_type="ctrl", ylabel="vx (m/s)")
            row = 1
            self.plot_uav_data(row, col, 1, data_type="ctrl", ylabel="vy (m/s)")
            row = 2
            self.plot_uav_data(row, col, 2, data_type="ctrl", ylabel="vz (m/s)")

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


def plot(self, pwm=False):
    """Logs entries for a single simulation step, of a single drone.
    copied from: https://github.com/utiasDSL/gym-pybullet-drones/blob/main/gym_pybullet_drones/utils/Logger.py

    Parameters
    ----------
    pwm : bool, optional
        If True, converts logged RPM into PWM values (for Crazyflies).

    """
    #### Loop over colors and line styles ######################
    plt.rc(
        "axes",
        prop_cycle=(
            cycler("color", ["r", "g", "b", "y"])
            + cycler("linestyle", ["-", "--", ":", "-."])
        ),
    )
    fig, axs = plt.subplots(10, 2)
    t = np.arange(
        0, self.timestamps.shape[1] / self.LOGGING_FREQ_HZ, 1 / self.LOGGING_FREQ_HZ
    )

    #### Column ################################################
    col = 0

    #### XYZ ###################################################
    row = 0
    for j in range(self.NUM_DRONES):
        axs[row, col].plot(t, self.states[j, 0, :], label="drone_" + str(j))
    axs[row, col].set_xlabel("time")
    axs[row, col].set_ylabel("x (m)")

    row = 1
    for j in range(self.NUM_DRONES):
        axs[row, col].plot(t, self.states[j, 1, :], label="drone_" + str(j))
    axs[row, col].set_xlabel("time")
    axs[row, col].set_ylabel("y (m)")

    row = 2
    for j in range(self.NUM_DRONES):
        axs[row, col].plot(t, self.states[j, 2, :], label="drone_" + str(j))
    axs[row, col].set_xlabel("time")
    axs[row, col].set_ylabel("z (m)")

    #### RPY ###################################################
    row = 3
    for j in range(self.NUM_DRONES):
        axs[row, col].plot(t, self.states[j, 6, :], label="drone_" + str(j))
    axs[row, col].set_xlabel("time")
    axs[row, col].set_ylabel("r (rad)")
    row = 4
    for j in range(self.NUM_DRONES):
        axs[row, col].plot(t, self.states[j, 7, :], label="drone_" + str(j))
    axs[row, col].set_xlabel("time")
    axs[row, col].set_ylabel("p (rad)")
    row = 5
    for j in range(self.NUM_DRONES):
        axs[row, col].plot(t, self.states[j, 8, :], label="drone_" + str(j))
    axs[row, col].set_xlabel("time")
    axs[row, col].set_ylabel("y (rad)")

    #### Ang Vel ###############################################
    row = 6
    for j in range(self.NUM_DRONES):
        axs[row, col].plot(t, self.states[j, 9, :], label="drone_" + str(j))
    axs[row, col].set_xlabel("time")
    axs[row, col].set_ylabel("wx")
    row = 7
    for j in range(self.NUM_DRONES):
        axs[row, col].plot(t, self.states[j, 10, :], label="drone_" + str(j))
    axs[row, col].set_xlabel("time")
    axs[row, col].set_ylabel("wy")
    row = 8
    for j in range(self.NUM_DRONES):
        axs[row, col].plot(t, self.states[j, 11, :], label="drone_" + str(j))
    axs[row, col].set_xlabel("time")
    axs[row, col].set_ylabel("wz")

    #### Time ##################################################
    row = 9
    axs[row, col].plot(t, t, label="time")
    axs[row, col].set_xlabel("time")
    axs[row, col].set_ylabel("time")

    #### Column ################################################
    col = 1

    #### Velocity ##############################################
    row = 0
    for j in range(self.NUM_DRONES):
        axs[row, col].plot(t, self.states[j, 3, :], label="drone_" + str(j))
    axs[row, col].set_xlabel("time")
    axs[row, col].set_ylabel("vx (m/s)")
    row = 1
    for j in range(self.NUM_DRONES):
        axs[row, col].plot(t, self.states[j, 4, :], label="drone_" + str(j))
    axs[row, col].set_xlabel("time")
    axs[row, col].set_ylabel("vy (m/s)")
    row = 2
    for j in range(self.NUM_DRONES):
        axs[row, col].plot(t, self.states[j, 5, :], label="drone_" + str(j))
    axs[row, col].set_xlabel("time")
    axs[row, col].set_ylabel("vz (m/s)")

    #### RPY Rates #############################################
    row = 3
    for j in range(self.NUM_DRONES):
        rdot = np.hstack(
            [
                0,
                (self.states[j, 6, 1:] - self.states[j, 6, 0:-1])
                * self.LOGGING_FREQ_HZ,
            ]
        )
        axs[row, col].plot(t, rdot, label="drone_" + str(j))
    axs[row, col].set_xlabel("time")
    axs[row, col].set_ylabel("rdot (rad/s)")
    row = 4
    for j in range(self.NUM_DRONES):
        pdot = np.hstack(
            [
                0,
                (self.states[j, 7, 1:] - self.states[j, 7, 0:-1])
                * self.LOGGING_FREQ_HZ,
            ]
        )
        axs[row, col].plot(t, pdot, label="drone_" + str(j))
    axs[row, col].set_xlabel("time")
    axs[row, col].set_ylabel("pdot (rad/s)")
    row = 5
    for j in range(self.NUM_DRONES):
        ydot = np.hstack(
            [
                0,
                (self.states[j, 8, 1:] - self.states[j, 8, 0:-1])
                * self.LOGGING_FREQ_HZ,
            ]
        )
        axs[row, col].plot(t, ydot, label="drone_" + str(j))
    axs[row, col].set_xlabel("time")
    axs[row, col].set_ylabel("ydot (rad/s)")

    ### This IF converts RPM into PWM for all drones ###########
    #### except drone_0 (only used in examples/compare.py) #####
    for j in range(self.NUM_DRONES):
        for i in range(12, 16):
            if pwm and j > 0:
                self.states[j, i, :] = (self.states[j, i, :] - 4070.3) / 0.2685

    #### RPMs ##################################################
    row = 6
    for j in range(self.NUM_DRONES):
        axs[row, col].plot(t, self.states[j, 12, :], label="drone_" + str(j))
    axs[row, col].set_xlabel("time")
    if pwm:
        axs[row, col].set_ylabel("PWM0")
    else:
        axs[row, col].set_ylabel("RPM0")
    row = 7
    for j in range(self.NUM_DRONES):
        axs[row, col].plot(t, self.states[j, 13, :], label="drone_" + str(j))
    axs[row, col].set_xlabel("time")
    if pwm:
        axs[row, col].set_ylabel("PWM1")
    else:
        axs[row, col].set_ylabel("RPM1")
    row = 8
    for j in range(self.NUM_DRONES):
        axs[row, col].plot(t, self.states[j, 14, :], label="drone_" + str(j))
    axs[row, col].set_xlabel("time")
    if pwm:
        axs[row, col].set_ylabel("PWM2")
    else:
        axs[row, col].set_ylabel("RPM2")
    row = 9
    for j in range(self.NUM_DRONES):
        axs[row, col].plot(t, self.states[j, 15, :], label="drone_" + str(j))
    axs[row, col].set_xlabel("time")
    if pwm:
        axs[row, col].set_ylabel("PWM3")
    else:
        axs[row, col].set_ylabel("RPM3")

    #### Drawing options #######################################
    for i in range(10):
        for j in range(2):
            axs[i, j].grid(True)
            axs[i, j].legend(loc="upper right", frameon=True)
    fig.subplots_adjust(
        left=0.06, bottom=0.05, right=0.99, top=0.98, wspace=0.15, hspace=0.0
    )
    if self.COLAB:
        plt.savefig(os.path.join("results", "output_figure.png"))
    else:
        plt.show()
