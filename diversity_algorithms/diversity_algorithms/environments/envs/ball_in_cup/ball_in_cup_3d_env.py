import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

steps_done = 0
monitor_rate = 1000
total_rew_seen = 0

class BallInCup3dEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, relative_obs=True, dense=False, random_init=False, verbose=False):
        self.max_steps = 200
        self.steps = 0
        self.dense = dense
        self.random_init = random_init
        self.verbose = verbose
        self.relative_obs = relative_obs

        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, "/Users/chenu/Desktop/PhD/github/trajectory_AE/envs/ball_in_cup/assets/ball_in_cup_3d.xml", 2)
        self.init_qpos = [0. ,0. ,0.   # initial cup position in cartesian space
                          ,0. ,0. ,0.] # initial ball posiiton in cartesian space

        self.init_qvel = [0. ,0. ,0.   # initial cup velocity
                          ,0. ,0. ,0.] # initial ball velocity

    def _is_done(self, info):
        done = False
        if self.in_target():
            done = True
        if self.steps >= self.max_steps:
            info["TimeLimit.truncated"] = not done
            done = True
        info["relative_ball_to_target_pos"] = self.ball_to_target()
        info["relative_ball_to_target_vel"] = self.sim.data.qvel.flat[:3] - self.sim.data.qvel.flat[3:] # target - ball vel (target vel == cup vel)
        return done

    def _monitoring(self, reward):
        global steps_done, total_rew_seen
        steps_done += 1
        total_rew_seen += reward
        if steps_done%monitor_rate == 0:
            print(f"{steps_done} steps done. Cumulated reward seen {total_rew_seen}")

    def step(self, a):
        reward_ctrl = -np.square(a).sum()
        task_reward = self.get_reward()
        reward = task_reward # + reward_ctrl

        self.do_simulation(a, self.frame_skip)
        self.steps += 1

        obs = self._get_obs()
        info = dict(task_reward=task_reward, reward_ctrl=reward_ctrl)
        done = self._is_done(info)

        if self.verbose:
            self._monitoring(reward)

        return obs, reward, done, info

    def viewer_setup(self):
        from mujoco_py.generated import const
        self.viewer._paused = True # start viewer paused
        self.viewer.cam.fixedcamid = 0 # start viewer at cam 0
        self.viewer.cam.type = const.CAMERA_FIXED # cam 0 is a fixed cam (-1 is free)

    def reset_model(self):
        self.steps = 0
        # qpos is [cup_x, cup_y, cup_z, ball_x, ball_y, ball_z]
        # qvel is analogous

        # Random initialization
        if self.random_init:
            qpos = (
                self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq)
                + self.init_qpos
            )

            qvel = self.init_qvel + self.np_random.uniform(
                low=-0.005, high=0.005, size=self.model.nv
            )
            qpos[:3] = self.init_qpos[:3] # still set cup pose to init
            qvel[:3] = self.init_qvel[:3] # still set cup vel to init
        else:
            qpos = np.array(self.init_qpos)
            qvel = np.array(self.init_qvel)

        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        if self.relative_obs:
            return np.concatenate((
                # self.sim.data.qpos.flat[:3] - self.sim.data.qpos.flat[3:], # cup - ball pose
                # self.sim.data.qvel.flat[:3] - self.sim.data.qvel.flat[3:],) # cup - ball vel
                self.get_site_pos("target") - self.get_site_pos("ball"), # target - ball pose
                self.sim.data.qvel.flat[:3] - self.sim.data.qvel.flat[3:],) # target - ball vel (target vel == cup vel)
            )
        else:
            return np.concatenate((
                self.sim.data.qpos.flat[:3], # cup pose
                self.sim.data.qvel.flat[:3], # cup vel
                self.sim.data.qpos.flat[3:], # ball pose
                self.sim.data.qvel.flat[3:],) # ball vel
            )

    def get_site_pos(self, site_name):
        if site_name == "cup":
            return self.sim.data.site_xpos[0]
        elif site_name == "target":
            return self.sim.data.site_xpos[1]
        elif site_name == "ball":
            return self.sim.data.site_xpos[2]
        raise ValueError(f"{site_name} is not a valid option for get_site_pos method. Valid options are cup/target/ball")

    def ball_to_target(self):
        """Returns the vector from the ball to the target."""
        # target = self.sim.data.site_xpos[1, [0, 1, 2]]
        # ball = self.sim.data.site_xpos[2, [0, 1, 2]]
        target = self.get_site_pos("target")
        ball = self.get_site_pos("ball")
        return target - ball

    def in_target(self):
        """Returns 1 if the ball is in the target, 0 otherwise."""
        ball_to_target = abs(self.ball_to_target())
        target_size = self.sim.model.site_size[1, [0, 1, 2]]
        ball_size = self.sim.model.geom_size[2, 0]
        return float(all(ball_to_target < target_size - ball_size))

    def get_reward(self):
        """Returns a dense or sparse reward."""
        if self.dense:
            dist = self.get_site_pos("target") - self.get_site_pos("ball")
            reward = -np.linalg.norm(dist)
        else:
            reward = self.in_target()
        return reward

if __name__ == '__main__':
    import gym
    import mb_ge
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    env = DummyVecEnv([lambda: gym.make("BallInCup3d-v0", verbose=True, dense=False)])
    # Automatically normalize the input features and reward
    env = VecNormalize(env, norm_obs=True, norm_reward=True,
                       clip_obs=10.)

    # PPO example, PPO is on-policy
    # from stable_baselines3 import PPO
    # model = PPO("MlpPolicy", env, verbose=0)

    # model.learn(total_timesteps=100000)


    # DDPG example, DDPG is off-policy, expect longer computation time
    from stable_baselines3 import DDPG
    from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

    # The noise objects for DDPG
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1)
    model.learn(total_timesteps=100000, log_interval=10)

    # SAC example, SAC is off-policy, expect longer computation time
    # from stable_baselines3 import SAC

    # model = SAC("MlpPolicy", env, verbose=1)
    # model.learn(total_timesteps=10000, log_interval=4)

    # TD3 example, TD3 is off-policy, expect longer computation time
    # from stable_baselines3 import TD3
    # from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

    # # The noise objects for TD3
    # n_actions = env.action_space.shape[-1]
    # action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    # model = TD3("MlpPolicy", env, action_noise=action_noise, verbose=1)
    # model.learn(total_timesteps=10000, log_interval=10)

    # Visualization Loop
    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()

    env.close()
