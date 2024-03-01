import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from copy import deepcopy
from rl_mus.agents.agents import UavCtrlType
from rl_mus.utils.math_utils import calc_cum_sum


class BaseLogger(object):
    def __init__(self, num_uavs=1, log_freq=10) -> None:
        self.num_uavs = num_uavs

        # used for converting the uav_id to array index
        self.uav_ids = {}
        self._data = None
        self._num_samples = 0
        self._uav_counter = 0
        self.log_freq = log_freq

        for uav_id in range(self.num_uavs):
            self.add_uav(uav_id)

    def add_uav(self, uav_id):
        self.uav_ids[uav_id] = self._uav_counter
        self._uav_counter += 1

    def log(self):
        raise NotImplemented()

    @property
    def num_samples(self):
        raise NotImplemented()

    @property
    def data(self):
        return self._data


class UavLogger(BaseLogger):
    def __init__(self, num_uavs=1, log_freq=240, ctrl_type=UavCtrlType.VEL) -> None:
        self.ctrl_type = ctrl_type

        super().__init__(num_uavs=num_uavs, log_freq=log_freq)

        # used for converting the uav_id to array index
        self._data = {}
        self._data["log"] = [{"action": [], "state": []} for i in range(self.num_uavs)]

    def log(self, uav_id, state, action):
        array_idx = self.uav_ids[uav_id]
        self._data["log"][array_idx]["state"].append(state)
        self._data["log"][array_idx]["action"].append(action)

    @property
    def num_samples(self):
        self._num_samples = self._data["log"][0]["state"].shape[0]

        return self._num_samples

    def plot(self, title="", plt_action=False, plt_target=False):
        if self.num_uavs > 1 and self.num_uavs <= 4:
            colors = ["r", "g", "b", "y"]
            linestyle = ["-", "--", ":", "-."]

            c = [colors[i] for i in range(self.num_uavs)]
            l = [linestyle[i] for i in range(self.num_uavs)]
            plt.rc(
                "axes",
                prop_cycle=(cycler("color", c) + cycler("linestyle", l)),
            )

        num_rows = 6
        num_cols = 2

        self.fig, self.axs = plt.subplots(
            num_rows, num_cols, sharex=True, figsize=(14, 12)
        )
        self.fig.suptitle(title)

        # convert data to numpy arrays
        self._data_np = deepcopy(self._data)

        for uav_id, idx in self.uav_ids.items():
            for key in self._data_np["log"][idx].keys():
                self._data_np["log"][idx][key] = np.array(
                    self._data_np["log"][idx][key]
                )

        self._num_samples = self._data_np["log"][0]["state"].shape[0]
        col = 0
        # x, y, z
        row = 0
        self.plot_uav_data(row, col, 0, ylabel="x (m)")
        row = 1
        self.plot_uav_data(row, col, 1, ylabel="y (m)")
        row = 2
        self.plot_uav_data(row, col, 2, ylabel="z (m)")

        if plt_target:
            row = 0
            self.plot_uav_data(row, col, 0, data_type="target", ylabel="x (m)")
            row = 1
            self.plot_uav_data(row, col, 1, data_type="target", ylabel="y (m)")
            row = 2
            self.plot_uav_data(row, col, 2, data_type="target", ylabel="z (m)")

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

        # TODO: can't get RPMS now without normalizing the observation space first
        # RPMS
        # col = 0
        # row = 6
        # self.plot_uav_data(row, col, 16, ylabel="RPM0")
        # row = 7
        # self.plot_uav_data(row, col, 17, ylabel="RPM1")

        # col = 1
        # row = 6
        # self.plot_uav_data(row, col, 18, ylabel="RPM2")
        # row = 7
        # self.plot_uav_data(row, col, 19, ylabel="RPM3")

        if plt_action:
            if self.ctrl_type == UavCtrlType.VEL:
                col = 1
                row = 0
                self.plot_uav_data(row, col, 0, data_type="action", ylabel="vx (m/s)")
                row = 1
                self.plot_uav_data(row, col, 1, data_type="action", ylabel="vy (m/s)")
                row = 2
                self.plot_uav_data(row, col, 2, data_type="action", ylabel="vz (m/s)")

            elif self.ctrl_type == UavCtrlType.POS:
                col = 0
                row = 0
                self.plot_uav_data(row, col, 0, data_type="action", ylabel="x (m)")
                row = 1
                self.plot_uav_data(row, col, 1, data_type="action", ylabel="y (m)")
                row = 2
                self.plot_uav_data(row, col, 2, data_type="action", ylabel="z (m)")
                row = 5
                self.plot_uav_data(
                    row, col, 3, data_type="action", ylabel="$\psi$ (rad)"
                )

            elif self.ctrl_type == UavCtrlType.RPM:
                # RPMS
                col = 0
                row = 6
                self.plot_uav_data(row, col, 0, data_type="action", ylabel="RPM0")
                row = 7
                self.plot_uav_data(row, col, 1, data_type="action", ylabel="RPM1")

                col = 1
                row = 6
                self.plot_uav_data(row, col, 2, data_type="action", ylabel="RPM2")
                row = 7
                self.plot_uav_data(row, col, 3, data_type="action", ylabel="RPM3")

        for row in range(num_rows):
            for col in range(num_cols):
                self.axs[row, col].grid(True)
                self.axs[row, col].legend(loc="upper right", frameon=True)

        self.fig.subplots_adjust(hspace=0)

        plt.show()

    def plot_uav_data(self, row, col, data_idx, data_type="state", ylabel=""):
        t = np.arange(self._num_samples) / self.log_freq
        for uav_id, idx in self.uav_ids.items():
            data = self._data_np["log"][idx][data_type][:, data_idx]
            self.axs[row, col].plot(t, data, label=f"uav_{uav_id}")
        self.axs[row, col].set_xlabel("t (s)")
        self.axs[row, col].set_ylabel(ylabel)


class EnvLogger(UavLogger):
    def __init__(self, num_uavs=1, log_config={}) -> None:

        self._log_config = log_config
        log_freq = self._log_config.setdefault("log_freq", 10)
        uav_ctrl_type = self._log_config.setdefault("uav_ctrl_type", UavCtrlType.VEL)
        self._env_freq = self._log_config["env_freq"]
        self._log_step = -1

        super().__init__(num_uavs=num_uavs, log_freq=log_freq, ctrl_type=uav_ctrl_type)

        self._log_step_skip = int(self._env_freq / self.log_freq)
        self.parse_log_config()

    def parse_log_config(self):
        self._obs_items = self._log_config.setdefault("obs_items", ["state"])
        self._log_reward = self._log_config.setdefault("log_reward", False)
        self._info_items = self._log_config.setdefault("info_items", [])

        self._data = {}
        self._data["eps_time_step"] = []
        self._data["eps_num"] = []
        self._data["eps_tot_sim_time"] = []
        self._data["eps_tot_real_time"] = []
        data_dictionary = {}

        for key in self._obs_items:
            data_dictionary[key] = []
        for key in self._info_items:
            data_dictionary[key] = []
        if self._log_reward:
            data_dictionary["reward"] = []
            data_dictionary["cum_reward"] = []
        data_dictionary["action"] = []

        self._data["log"] = [deepcopy(data_dictionary) for i in range(self.num_uavs)]

    def log(self, eps_num, info, obs, reward, action):

        self._log_step += 1
        if not (self._log_step % self._log_step_skip) == 0:
            return
        self._data["eps_time_step"].append(self._log_step / self.log_freq)
        self._data["eps_num"].append(eps_num)

        for uav_id, value in obs.items():
            array_idx = self.uav_ids[uav_id]

            for k, v in value.items():
                if k in self._obs_items:
                    self.data["log"][array_idx][k].append(v.tolist())

            for k, v in info[uav_id].items():
                if k in self._info_items:
                    self.data["log"][array_idx][k].append(v)

            if self._log_reward:
                self.data["log"][array_idx]["reward"].append(reward[uav_id])

            self.data["log"][array_idx]["action"].append(action[uav_id].tolist())

    def log_eps_time(self, sim_time, real_time):
        self._data["eps_tot_sim_time"].append(sim_time)
        self._data["eps_tot_real_time"].append(real_time)

    @property
    def num_samples(self):
        return len(self.data["eps_num"])

    def plot_env(self, title=""):

        if self.num_uavs > 1 and self.num_uavs <= 4:
            colors = ["r", "g", "b", "y"]
            linestyle = ["-", "--", ":", "-."]

            c = [colors[i] for i in range(self.num_uavs)]
            l = [linestyle[i] for i in range(self.num_uavs)]
            plt.rc(
                "axes",
                prop_cycle=(cycler("color", c) + cycler("linestyle", l)),
            )

        num_rows = 8
        num_cols = 1

        self.fig, self.axs = plt.subplots(
            num_rows, num_cols, sharex=True, figsize=(14, 12)
        )
        self.fig.suptitle(title)

        # convert data to numpy arrays
        self._data_np = deepcopy(self._data)
        for uav_id, idx in self.uav_ids.items():
            for key in self._data_np["log"][idx].keys():
                self._data_np["log"][idx][key] = np.array(
                    self._data_np["log"][idx][key]
                )

            self._data_np["log"][idx]["cum_reward"] = calc_cum_sum(self._data_np["log"][idx]['reward'], self._data_np["eps_num"])

        self._num_samples = self._data_np["log"][0]["state"].shape[0]
        col = 0
        row = 0
        self.plot_info_data(row, data_type="uav_target_reached", ylabel="tgt reached")
        row += 1
        self.plot_info_data(row, data_type="reward", ylabel="reward")
        # TODO: calculate cumulative reward
        # TODO: calculate discounted reward
        row += 1 
        self.plot_info_data(row, data_type="cum_reward", ylabel="cum reward")
        row += 1
        self.plot_info_data(row, data_type="uav_done_dt", ylabel="done dt")
        row += 1
        self.plot_info_data(row, data_type="uav_collision", ylabel="uav_col")
        row += 1
        self.plot_info_data(row, data_type="obstacle_collision", ylabel="ncfo_col")
        row += 1
        self.plot_info_data(
            row,
            data_type="uav_rel_dist",
            ylabel="$\parallel \Delta \mathbf{r} \parallel$",
        )
        row += 1
        self.plot_info_data(
            row,
            data_type="uav_rel_vel",
            ylabel="$\parallel \Delta \mathbf{v} \parallel$",
        )

        for row in range(num_rows):
            self.axs[row].grid(True)
            self.axs[row].legend(loc="upper right", frameon=True)

        self.fig.subplots_adjust(hspace=0)

        plt.show()

    def plot_info_data(
        self,
        row,
        data_type="reward",
        ylabel="",
    ):
        t = np.arange(self._num_samples) * self._log_step_skip / self._env_freq
        for uav_id, idx in self.uav_ids.items():
            data = self._data_np["log"][idx][data_type]
            self.axs[row].plot(t, data, label=f"uav_{uav_id}")
        self.axs[row].set_xlabel("t (s)")
        self.axs[row].set_ylabel(ylabel)


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
