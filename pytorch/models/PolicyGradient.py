import numpy as np
from torch.nn import Sequential, Linear, Tanh, Module, Softmax, ReLU
from torch.nn.functional import cross_entropy, softmax, one_hot
from torch.optim import Adam
import torch.nn as nn
import torch
from os.path import join
from torch.distributions import Categorical

# reproducible
np.random.seed(1)
torch.manual_seed(543)  # 策略梯度算法方差很大，设置seed以保证复现性


class PolicyGradientLoss(Module):
    def __init__(self, gamma):
        super().__init__()

        self.gamma = gamma

    def forward(self, weight, ep_as, ep_rs):
        # print(x, y)
        discounted_ep_rs = torch.zeros_like(ep_rs)
        running_add = 0
        # 注意这里是从后往前算的，所以式子还不太一样。算出每一步的状态价值
        # 前面的价值的计算可以利用后面的价值作为中间结果，简化计算；从前往后也可以
        for t in reversed(range(0, len(ep_rs))):
            running_add = running_add * self.gamma + ep_rs[t]
            discounted_ep_rs[t] = running_add

        discounted_ep_rs -= torch.mean(discounted_ep_rs)  # 减均值
        neg_log_prob = cross_entropy(input=weight, target=ep_as, reduction='none')
        loss = torch.mean(neg_log_prob * discounted_ep_rs)
        return loss


class PolicyGradient(Module):
    def __init__(
            self,
            n_actions,
            n_features,
    ):
        super(PolicyGradient, self).__init__()
        self.n_actions = n_actions
        self.n_features = n_features
        self.model = Sequential(
            Linear(self.n_features, 20),
            ReLU(),
            Linear(20, self.n_actions)
            # Softmax(dim=0)
        )

    def forward(self, x):
        return self.model(x)

    def initialize_weights(self):
        for m in self.modules():
            nn.init.normal_(m.weight.data, 0, 0.1)
            nn.init.constant_(m.bias.data, 0.01)
            # m.bias.data.zero_()


class Agent:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.95,
            device='cpu'
    ):
        self.model = PolicyGradient(n_actions, n_features).to(device)
        self.gamma = reward_decay
        self.optim = Adam(self.model.parameters(), lr=learning_rate)
        self.cost = PolicyGradientLoss(self.gamma)
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []
        self.device = device

    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def choose_action(self, observation):
        prob_weights = self.model(torch.FloatTensor(observation).to(self.device))
        # print(prob_weights)
        with torch.no_grad():
            prob_weights = softmax(prob_weights, dim=0).data.numpy()
        action = np.random.choice(range(prob_weights.shape[0]),
                                  p=prob_weights)  # select action w.r.t the actions prob
        return action

    def learn(self):
        ret = self.model(torch.FloatTensor(np.array(self.ep_obs)).to(self.device))

        sum_reward = sum(self.ep_rs)
        loss = self.cost(ret, torch.LongTensor(self.ep_as).to(self.device),
                         torch.FloatTensor(self.ep_rs).to(self.device))
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []
        return sum_reward

    def save_checkpoint(self, epoch, directory):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optim.state_dict(),
        }, join(directory, str(epoch) + '.pth'))

    def save_model(self, path):
        torch.save(self.model, path)

    def load_checkPoint(self, url):
        checkpoint = torch.load(url)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optim.load_state_dict(checkpoint['optimizer_state_dict'])
        # epoch = checkpoint['epoch']

