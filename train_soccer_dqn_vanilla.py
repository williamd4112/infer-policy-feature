from soccer_env import SoccerEnv, wrap_dqn_for_soccer

from baselines.common.atari_wrappers_deprecated import ScaledFloatFrame

from experiment import deepq

def main():
    env = SoccerEnv(frameskip=1)
    env = ScaledFloatFrame(wrap_dqn_for_soccer(env))
    model = deepq.model

    act = deepq.learn(
        env,
        q_func=model,
        lr=1e-3,
        batch_size=64,
        max_timesteps=200000,
        buffer_size=10000,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        train_freq=4,
        learning_starts=10000,
        target_network_update_freq=1000,
        gamma=0.99,
        prioritized_replay=False
    )
    act.save("soccer_model_dqn_vanilla.pkl")
    env.close()


if __name__ == '__main__':
    main()
