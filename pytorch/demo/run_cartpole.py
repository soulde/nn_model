import gym
import argparse
from pytorch.models.PolicyGradient import Agent


def params():
    parser = argparse.ArgumentParser('RL demo')
    parser.add_argument('-e', '--episode', default=3000)
    parser.add_argument('-lr', "--learning_rate", default=0.01)
    parser.add_argument('--gamma', default=0.95)
    parser.add_argument('-p', '--path', default=0.95)
    return parser.parse_args()


def main(args):
    env = gym.make('CartPole-v1')
    env = env.unwrapped
    agent = Agent(env.action_space.n, env.observation_space.shape[0], args.learning_rate, args.gamma, device='cpu')

    for i in range(args.episode):
        observation = env.reset()[0]
        # print(observation)
        if i > 400 and i % 20 == 0:
            agent.save_model(args.path)
        while True:
            env.render()
            action = agent.choose_action(observation)
            observation_, reward, done, truncated, info = env.step(action)

            agent.store_transition(observation, action, reward)
            observation = observation_
            if done or truncated:
                print(info)
                rewards = agent.learn()
                print('Episode', i, 'reward', rewards)
                print(truncated)
                break


if __name__ == '__main__':
    args = params()
    main(args)
