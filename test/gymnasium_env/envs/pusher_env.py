"""
python -m rl_zoo3.enjoy --algo ppo --env gymnasium_env/PingPongEnv-v0 -f logs/ --exp-id 0
python -m rl_zoo3.train --algo ppo --env gymnasium_env/PingPongEnv-v0 --tensorboard-log ./logs/ppo/tensorboard_log --conf-file pingpong.yml -P
"""

"""
qpos =
x ball position
y ball position
z ball position


"""
from pathlib import Path
from typing import Dict, Union

import numpy as np
from icecream import ic
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box


DEFAULT_CAMERA_CONFIG = {
    "distance": 2.0,
    "lookat": np.array((0.5,0,0)),
    "elevation": -40,
}


class PingPongEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }

    def __init__(
        self,
        #xml_file: str = "pusher_v5.xml",
        xml_file: str = str(Path(__file__).resolve().parents[3] / "urdf" / "braccioLight" / "pongace.xml"),
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
        self.step_counter = 0
        self._hitted = False # Flag per il controllo della collisione
        self._gravity = False
        self.stop = False
        self.count_hit = 0
        low = np.full(23, -np.inf)
        high = np.full(23, np.inf)

        low[:3] = [0,0,0]
        high[:3] = [10, 3, 6]
        observation_space = Box(low=low, high=high, dtype=np.float64)


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

        ball_pos = self.get_body_com("ball")
        truncation = False
        # ball position box limits
        limits = [[-1,8],[-2,2],[-1,6]]
        for i in range(3):
            if ball_pos[i] < limits[i][0] or ball_pos[i] > limits[i][1]:
                # if the arm hit the ball not in the goal reward is negative
                reward -= 100
                truncation = True
                break



        self.step_count += 1  # increase step counter
        if self.stop:
            truncation = True
            self.stop = False
        if self.step_count > 100:
            truncation = True
        return observation, reward, False, truncation, info

    def _get_rew(self, action):
        vec_1 = self.get_body_com("ball") - self.get_body_com("racket_center")
        vec_2 = self.get_body_com("ball") - self.get_body_com("goal")

        reward_near = -np.linalg.norm(vec_1) * self._reward_near_weight
        reward_dist = -np.linalg.norm(vec_2) * self._reward_dist_weight
        reward_ctrl = -np.square(action).sum() * self._reward_control_weight


        reward = reward_ctrl + reward_near +reward_dist
        if np.linalg.norm(vec_1) < 0.3:
            self.count_hit += 1
            reward += 100
        if np.linalg.norm(vec_2) < 1.5:
            reward += 10

        elbow_pos = self.get_body_com("racket_center")
        ball_pos = self.get_body_com("ball")
        goal_pos = self.get_body_com("goal")

        if np.abs(goal_pos[0] - ball_pos[0]) < 1 and np.abs(goal_pos[1] - ball_pos[1]) < 2.3 and np.abs(goal_pos[2] - ball_pos[2]) < 4.0:
            reward += 200
            print("Reward!")
            self.stop = True
        # Calcola le distanze
        #distance_elbow_to_object = np.linalg.norm(elbow_pos - ball_pos)
        #distance_object_to_goal = np.linalg.norm(ball_pos - goal_pos)
        # Calcola la variazione della distanza tra l'oggetto e il goal
        #if hasattr(self, 'previous_distance_object_to_goal'):
        #    delta_distance_object_to_goal = self.previous_distance_object_to_goal - distance_object_to_goal
        #else:
        #    delta_distance_object_to_goal = 0

        # Penalità per la staticità
        #joint_velocities = self.data.qvel
        #static_penalty =  np.sum(np.abs(joint_velocities)) * 0.1

        # Funzione di ricompensa: penalizza le distanze maggiori e premia la riduzione della distanza
        #reward = - (distance_elbow_to_object + distance_object_to_goal) + delta_distance_object_to_goal + static_penalty
        #reward = - distance_elbow_to_object * 10 - distance_object_to_goal * 10
        #reward = - distance_elbow_to_object  * 0.1

        # Aggiorna la distanza precedente
        #self.previous_distance_object_to_goal = distance_object_to_goal

        #if self._gravity:
        #    self.model.opt.gravity[:] = [0, 0, -9.81]

        #if distance_elbow_to_object < 0.3:
        #    self.count_hit += 1
        #    print("hit")
        #    reward += 75

        #
        #if distance_object_to_goal < 1.51:
        #    reward += 200

        # Informazioni aggiuntive per il debug
        reward_info = {
            #"distance_elbow_to_object": distance_elbow_to_object,
            #"distance_object_to_goal": distance_object_to_goal,
            #"delta_distance_object_to_goal": delta_distance_object_to_goal,
            "reward": reward,
            "hit_count": self.count_hit,
        }

        return reward, reward_info

    def reset_model(self):
        self.step_count = 0
        qpos = self.init_qpos
        qvel = self.init_qvel
        #pos_z = np.random.rand() * 3
        #pos_y = 0.8
        pos_y = np.random.rand() * 2
        #pos_y = 0
        pos_x = 6.5
        #pos_y = -0.5
        #pos_y = 1.10
        pos_z = 4.5
        qpos[0] = pos_x
        qpos[1] = pos_y
        qpos[2] = pos_z
        #self.model.opt.gravity[:] = [0, 0, 0]

        qvel[0] = -5
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        obs = np.concatenate(
            [
                self.data.qpos.flatten()[:7],#[:3],
                self.data.qvel.flatten()[:7],#[:3],
                self.get_body_com("racket_center"),
                self.get_body_com("ball"),
                self.get_body_com("goal"),
            ]
        )
        return obs / np.linalg.norm(obs)
