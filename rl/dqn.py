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

class DeepQ(nn.Module):
    def __init__(
            self,
            n_actions,
            n_features=512,
            hidden=128
    ):
        super(DeepQ, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x, actions_one_hot):
        # q_outputs
        x = self.layer(x)  # shape: (B, 10)
        self.q_outputs = x
        # x = x[torch.arange(len(x)), torch.argmax(actions_one_hot, -1)]
        x = actions_one_hot.mul(x) # shape: (B, 10)
        x = torch.sum(x, dim=1, keepdims=True) # shape: (B, 1)
        return x


class DeepQResnet(nn.Module):
    def __init__(self):
        super(DeepQResnet, self).__init__()
        # self.conv = nn.Conv2d(1, 3, kernel_size=1)
        model = models.resnet50(pretrained=True)
        model.fc = nn.Identity()
        self.resnet = model

    def forward(self, x): 
        # x= self.conv(x)
        x= self.resnet(x)
        return x


class MnistDeepQ(nn.Module):
    def __init__(
            self,
            n_actions,
            n_features=512,
            hidden=128
    ):
        super(MnistDeepQ, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(512, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions)
        )

    def forward(self, x, actions_one_hot):
        # q_outputs
        x = self.layer(x)  # shape: (B, 10)
        self.q_outputs = x
        # x = x[torch.arange(len(x)), torch.argmax(actions_one_hot, -1)]
        x = actions_one_hot.mul(x) # shape: (B, 10)
        x = torch.sum(x, dim=1, keepdims=True) # shape: (B, 1)
        return x

Transition = namedtuple('Transition',
                        ('state', 'action', 'old_q', 'reward', 'next_state'))

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
            learning_rate=0.001,
            gamma=0.9,
            ep_max=1,
            ep_min=0.01,
            ep_decay=50,
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
        self.feature_extractor = feature_extractor.cuda() if feature_extractor is not None  else None
        self.eval_net, self.target_net = DeepQ(n_actions, n_features), DeepQ(n_actions, n_features)
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.target_net.parameters(), lr=learning_rate)

    # memory
    def remember(self, state, action, old_q, reward, next_state):
        self.memory.push(state, action, old_q, reward, next_state)

    def random_sample(self):
        return self.memory.sample(self.batch_size)

    def pre_remember(self, env):
        state = env.reset()
        for i in range(self.pre_remember_num):
            rd_action = env.sample_actions()
            next_state, reward = env.step(rd_action)
            self.remember(state, rd_action, 0, reward, next_state)
            state = next_state

    def epsilon_calc(self, step, ep_decay=0.0001, esp_total=1000):
        return max(self.epsilon_min, self.epsilon_max - (self.epsilon_max - self.epsilon_min) * step / esp_total)

    def epsilon_calc2(self, step):
        return self.epsilon_min + (self.epsilon_max - self.epsilon_min) * math.exp(-1. * step / self.ep_decay)

    def select_action(self, env, state, step, ep_decay=0.0001, ep_total=1000):
        # epsilon = self.epsilon_calc(step, ep_decay, ep_total)
        epsilon = self.epsilon_calc2(step)
        if np.random.rand() < epsilon:
            return env.sample_actions(), 0
        dummy_actions = torch.ones((1, self.n_actions)).cuda()
        if self.feature_extractor is not None:
            self.feature_extractor.eval()
            features = self.feature_extractor(state.reshape(1, *state.shape))
            q_values = self.eval_net(features, dummy_actions)
            q_outputs = self.eval_net.q_outputs
        else:
            q_values = self.eval_net(state.reshape(1, *state.shape), dummy_actions)
            q_outputs = self.eval_net.q_outputs
        return torch.argmax(q_outputs).cpu().detach().numpy(), torch.max(q_outputs).cpu().detach().numpy()  # 返回q值最大的action和对应的q值

    def predict(self, states):
        if self.feature_extractor is not None:
            self.feature_extractor.eval()
            states = self.feature_extractor(states)
        q_values = self.eval_net(states, torch.ones((len(states), self.n_actions)).cuda())
        q_outputs = self.eval_net.q_outputs
        return torch.argmax(q_outputs, axis=1).cpu().detach().numpy()

    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.eval_net.load_state_dict(self.target_net.state_dict())
        # 从记忆中i, i, d采样
        samples = self.random_sample()
        # 展开所有样本的相关数据
        # 这里next_states没用 因为和上一个state无关
        # TODO: 尝试构造具有关联性质的state
        batch = Transition(*zip(*samples))
        states = [x.unsqueeze(0) for x in batch.state]
        states, actions, old_q, rewards = (torch.cat(states), np.array(batch.action).reshape(-1, 1),
                                           np.array(batch.old_q).reshape(-1, 1),
                                           np.array(batch.reward).reshape(-1, 1))

        actions = torch.from_numpy(actions).reshape(-1)
        # print(actions)
        actions_one_hot = torch.nn.functional.one_hot(actions, self.n_actions).cuda()
        # print(actions_one_hot)

        # print(states.shape,actions.shape,old_q.shape,rewards.shape,actions_one_hot.shape)
        # 从actor获取下一个状态的q估计值 这里也没用 因为gamma=0 也就是不对bellman方程展开
        # inputs_ = [next_states,np.ones((replay_size,num_actions))]
        # qvalues = actor_q_model.predict(inputs_)

        # q = np.max(qvalues, axis=1, keepdims=True)
        q = 0
        # 应用Bellman方程对Q进行更新，将新的Q更新给critic（方程9）
        # q_estimate = (1 - alpha) * old_q + alpha * (rewards.reshape(-1, 1) + gamma * q)
        q_estimate = (rewards.reshape(-1, 1) + gamma * q)
        q_estimate = torch.from_numpy(q_estimate).float().to(device)
        # 训练估计模型
        if self.feature_extractor is not None:
            states = self.feature_extractor(states)
        q_val = self.target_net(states, actions_one_hot)
        loss = self.loss_fn(q_val, q_estimate)
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
    
    # (x_train, y_train), (x_test, y_test) = custom_load_mnist_data()  # tf.keras.datasets.mnist.load_data()
    # num_actions = len(set(y_test))
    # image_w, image_h = x_train.shape[1:]

    # x_train = x_train.reshape(*x_train.shape, 1)
    # x_test = x_test.reshape(*x_test.shape, 1)
    # # normalization
    # x_train = x_train/255.0
    # x_test = x_test/255.0

    # x_train = torch.from_numpy(x_train).float().to(device)
    # x_test = torch.from_numpy(x_test).float().to(device)
    # x_train = x_train.permute(0, 3, 1, 2)
    # x_test = x_test.permute(0, 3, 1, 2)
    # env = MnistEnvironment(x_train, y_train)

    # y_train = torch.from_numpy(y_train)
    # y_test = torch.from_numpy(y_test)
    # y_train_one_hot = torch.nn.functional.one_hot(y_train.to(torch.int64), num_classes=num_actions).to(device)
    # y_test_one_hot = torch.nn.functional.one_hot(y_test.to(torch.int64), num_classes=num_actions).to(device)

    # # test_image = x_train[0]
    # # plt.imshow(test_image.reshape(28, 28), 'gray')
    # # plt.show()
    # # plt.close()
    # print(x_train.shape)
    # print(x_test.shape)
    # print('example num of train set: {}'.format(len(x_train)))
    # print('example num of test set: {}'.format(len(x_test)))

    import time
    num_actions = 10
    replay_size = 128
    epoches = 1
    pre_train_num = 256
    gamma = 0.  # every state is i.i.d
    alpha = 0.5
    forward = 1
    epislon_total = 2018
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
    env = RandomEnvironment(x_train, y_train, num_actions)

    feature_extractor=DeepQResnet()
    for param in feature_extractor.parameters():
        param.requires_grad = False
    dqn = DQN(num_actions, n_features=512, feature_extractor=feature_extractor, gamma=gamma)
    dqn.eval_net.to(device)
    dqn.target_net.to(device)
    print(dqn.eval_net)
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
            action, q = dqn.select_action(env, state, steps_done, ep_total=epislon_total)
            eps = dqn.epsilon_calc2(steps_done)
            steps_done += 1
            # eps = dqn.epsilon_calc(epoch, esp_total=epislon_total)
            # play
            next_state, reward = env.step(action)
            # 加入到经验记忆中
            dqn.remember(state, action, q, reward, next_state)

            # 从记忆中采样回放，保证iid。实际上这个任务中这一步不是必须的。
            loss = dqn.learn()

            total_rewards += reward
            state = next_state
            
        reward_rec.append(total_rewards)

        pbar.set_description('R:{} L:{:.4f} T:{} P:{:.3f}'.format(total_rewards, loss,
                                                                  int(time.time() - epoch_start), eps))

    r5 = np.mean([reward_rec[i:i + 10] for i in range(0, len(reward_rec), 10)], axis=1)

    plt.plot(range(len(r5)), r5, c='b')
    plt.xlabel('iters')
    plt.ylabel('mean score')
    plt.show()
    plt.close()


    pred = dqn.predict(x_test)

    from sklearn.metrics import accuracy_score
    print('test accuracy: {}'.format(accuracy_score(y_test.cpu().detach().numpy(), pred)))
    print(y_test.cpu().detach().numpy())
    print(pred)
