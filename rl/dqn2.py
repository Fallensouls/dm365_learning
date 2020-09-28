import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.autograd import Variable
from torchvision import models
import random
from collections import namedtuple
from environment import RandomEnvironment
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cifar100
import math

class Resnet18(nn.Module):
    def __init__(self, n_actions=10):
        super(Resnet18, self).__init__()
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(512, n_actions)
        self.net = model
    
    def forward(self, x):
        x = self.net(x)
        return x

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN():
    def __init__(
            self,
            n_actions,
            n_features,
            feature_extractor,
            net,
            learning_rate=0.001,
            gamma=0.9,
            ep_max=1,
            ep_min=0.2,
            ep_decay=2,
            replace_target_iter=128,
            memory_size=10000,
            pre_remember=256,
            batch_size=64,
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon_max = ep_max
        self.epsilon_min = ep_min
        self.ep_decay = ep_decay
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.pre_remember_num = pre_remember
        self.batch_size = batch_size
        self.learn_step_counter = 0
        self.memory = ReplayMemory(self.memory_size)
        self.cost_his = []
        self.feature_extractor = feature_extractor.cuda() if feature_extractor is not None else None
        self.eval_net, self.target_net = net, net
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=learning_rate)

    # memory
    def remember(self, state, action, reward, next_state):
        self.memory.push(state, action, reward, next_state)

    def random_sample(self):
        return self.memory.sample(self.batch_size)

    def pre_remember(self, env):
        state = env.reset()
        for i in range(self.pre_remember_num):
            rd_action = env.sample_actions()
            next_state, reward = env.step(rd_action)
            self.remember(state, rd_action, reward, next_state)
            state = next_state

    def epsilon_calc(self, step, ep_decay=0.0001, esp_total=1000):
        return max(self.epsilon_min, self.epsilon_max - (self.epsilon_max - self.epsilon_min) * step / esp_total)

    def epsilon_calc2(self, step):
        return self.epsilon_min + (self.epsilon_max - self.epsilon_min) * math.exp(-1. * step / self.ep_decay)

    def select_action(self, env, state, step, ep_decay=0.0001, ep_total=1000):
        # epsilon = self.epsilon_calc(step, ep_decay, ep_total)
        epsilon = self.epsilon_calc2(step)
        if np.random.rand() < epsilon:
            return env.sample_actions()    
        self.eval_net.eval()
        q_outputs = self.eval_net(state.reshape(1, *state.shape))
        return q_outputs.max(1)[1].view(1, 1)

    def predict(self, states):
        if self.feature_extractor is not None:
            self.feature_extractor.eval()
            states = self.feature_extractor(states)
        self.eval_net.eval()
        q_outputs = self.eval_net(states)
        return torch.argmax(q_outputs, axis=1).cpu().detach().numpy()

    def learn(self):
        if len(self.memory) < self.batch_size:
            return 0

        if self.learn_step_counter % self.replace_target_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        # 从记忆中i, i, d采样
        samples = self.random_sample()
        # 展开所有样本的相关数据
        # 这里next_states没用 因为和上一个state无关
        # TODO: 尝试构造具有关联性质的state
        batch = Transition(*zip(*samples))
        states = [x.unsqueeze(0) for x in batch.state]
        states = torch.cat(states)
        action_batch = torch.cat(batch.action)
        reward_batch = np.array(batch.reward).reshape(-1, 1)
        
        # next_state = [x.unsqueeze(0) for x in batch.next_state]
        # non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
        #                                   batch.next_state)), device=device, dtype=torch.bool)

        # non_final_next_states = torch.cat([s.unsqueeze(0) for s in batch.next_state
        #                                         if s is not None])
        # # print(states.shape)
        # print(non_final_next_states.shape)
        # next_state_values = torch.zeros(self.batch_size).cuda()
        # next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # print(next_state_values.shape)
        self.eval_net.train()
        state_action_values = self.eval_net(states).gather(1, action_batch)
        reward = torch.from_numpy(reward_batch).float().cuda()
        expected_state_action_values = reward
        # print(expected_state_action_values.shape)
        # print(reward.squeeze(1).shape)
        # print('q_output:{}'.format(self.eval_net.q_outputs))
        # print('q_val:{}'.format(q_val))
        # print('q_estimate:{}'.format(q_estimate))
        loss = self.loss_fn(state_action_values, expected_state_action_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        cost = loss.item()

        self.cost_his.append(cost)
        self.learn_step_counter += 1

        return loss


    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()


def custom_load_mnist_data(file_path='mnist.npz'):
    """自定义数据读取器"""
    file_data = np.load(file_path)
    x_train, y_train = file_data['x_train'], file_data['y_train']
    x_test, y_test = file_data['x_test'], file_data['y_test']
    file_data.close()
    return (x_train, y_train), (x_test, y_test)


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    import time
    num_actions = 10
    replay_size = 128
    epoches = 100
    pre_train_num = 256
    gamma = 0.9
    alpha = 0.9
    forward = 512
    epislon_total = 5120
    every_copy_step = 300
    total_rewards = 0
    reward_rec = []

    train_loader, test_loader = cifar100.load_cifar10()
    for batch_index, (images, labels) in enumerate(train_loader):
        x_train = images.cuda()
        y_train = labels.cuda()
    for batch_index, (images, labels) in enumerate(test_loader):
        x_test = images.cuda()
        y_test = labels.cuda()
    print(x_train.shape)
    print(x_test.shape)
    env = RandomEnvironment(x_train, y_train, num_actions)

    net = Resnet18()
    dqn = DQN(num_actions, n_features=512, feature_extractor=None, gamma=gamma, net=net)
    dqn.eval_net.to(device)
    dqn.target_net.to(device)
    # 填充初始经验池
    dqn.pre_remember(env)
    
    pbar = tqdm(range(1, epoches+1))

    steps_done = 0
    # 训练优化过程
    state = env.reset()
    for epoch in pbar:
        total_rewards = 0
        epoch_start = time.time()
        for step in range(forward):
            # 对每个状态使用epsilon_greedy选择
            action = dqn.select_action(env, state, steps_done, ep_total=epislon_total)
            eps = dqn.epsilon_calc2(steps_done)
            steps_done += 1
            # eps = dqn.epsilon_calc(steps_done, esp_total=epislon_total)
            
            next_state, reward = env.step(action)
            # 加入到经验记忆中
            dqn.remember(state, action, reward, next_state)

            loss = dqn.learn()

            total_rewards += reward
            state = next_state
           
            pbar.set_description('R:{} L:{:.4f} T:{} P:{:.3f}'.format(total_rewards, loss,
                                                                    int(time.time() - epoch_start), eps))
        from sklearn.metrics import accuracy_score
        pred = dqn.predict(x_test)
        print('test accuracy: {}'.format(accuracy_score(y_test.cpu().detach().numpy(), pred)))
        print(y_test.cpu().detach().numpy())
        print(pred)
        reward_rec.append(total_rewards)


    r5 = np.mean([reward_rec[i:i + 10] for i in range(0, len(reward_rec), 10)], axis=1)

    plt.plot(range(len(r5)), r5, c='b')
    plt.xlabel('iters')
    plt.ylabel('mean score')
    plt.show()
    plt.close()


    # from sklearn.metrics import accuracy_score
    # print('test accuracy: {}'.format(accuracy_score(y_test.cpu().detach().numpy(), pred)))
    # print(y_test.cpu().detach().numpy())
    # print(pred)
