from rl_mus.envs.rl_mus import RlMus
from rl_mus.envs.rl_mus_ttc import RlMusTtc
from ray.tune.registry import register_env


register_env(
    "rl-mus-v0",
    lambda env_config: RlMus(env_config=env_config),
)

register_env(
    "rl-mus-ttc-v0",
    lambda env_config: RlMusTtc(env_config=env_config),
)
