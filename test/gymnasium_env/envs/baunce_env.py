__credits__ = ["Kallinteris-Andreas"]

from typing import Dict, Union

import numpy as np
from icecream import ic
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box


DEFAULT_CAMERA_CONFIG = {
    "distance": 8.0,
    "lookat": np.array((3.0, 0.0, 1.0)),
    "elevation": -40,
}


class BaunceEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }

    def __init__(
        self,
        xml_file: str = "/Users/francesco/Desktop/pingpong/urdf/braccioLight/test3.xml",
        frame_skip: int = 5,
        default_camera_config: Dict[str, Union[float, int]] = DEFAULT_CAMERA_CONFIG,
        reward_near_weight: float = 0.5,
        reward_dist_weight: float = 1,
        reward_control_weight: float = 0.1,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            xml_file,
            frame_skip,
            default_camera_config,
            reward_near_weight,
            reward_dist_weight,
            reward_control_weight,
            **kwargs,
        )
        self._reward_near_weight = reward_near_weight
        self._reward_control_weight = reward_control_weight
        self._reward_dist_weight = reward_dist_weight
        
        
        
        observation_space = Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64)

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=observation_space,
            default_camera_config=default_camera_config,
            **kwargs,
        )

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",

            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

    def step(self, action):
        self.do_simulation(action, self.frame_skip)

        observation = self._get_obs()
        reward, reward_info = self._get_rew(action)
        info = reward_info

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return observation, reward, False, False, info

    def _get_rew(self, action):
        reward = 0
        reward_info = {}

        return reward, reward_info

    def reset_model(self):
        qpos = self.init_qpos
        
        qvel = self.init_qvel
        #qpos[2] = np.random.randint(3)
        #qpos[1] = np.random.randint(-2,2)
        qvel[0] = -1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate(
            [
                self.data.qpos.flatten()[:3],
                self.data.qvel.flatten()[:3],
                self.get_body_com("object"),
            ]
        )