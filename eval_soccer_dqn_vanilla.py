import gym

from baselines import deepq
from baselines.common.atari_wrappers_deprecated import ScaledFloatFrame
 
from soccer_env import SoccerEnv, wrap_dqn_for_soccer

def main():
    env = SoccerEnv(frameskip=1)
    env = ScaledFloatFrame(wrap_dqn_for_soccer(env))

    act = deepq.load("soccer_model.pkl")

    while True:
        obs, done = env.reset(), False
        episode_rew = 0
        while not done:
            env.render()
            obs, rew, done, _ = env.step(act(obs[None])[0])
            episode_rew += rew
        print("Episode reward", episode_rew)


if __name__ == '__main__':
    main()
