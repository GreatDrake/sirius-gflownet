import matplotlib.pyplot as plt
import torch
from math import log
from collections import OrderedDict
from itertools import product
import numpy
import itertools
import math
from IPython.display import clear_output
from matplotlib import pyplot as plt

#######################################################################################

N = 6
p = list(range(N))
print(p)
op = 1000000
batch = 1
hidden_size = 100
lr = 0.0001
lr_Z = 0.1


#######################################################################################


def transfer(state):
        global N
        res = [float(0)] * N * N
        for i in range(0, N):
                res[state[i] + i * N] = float(1)
        return res


# def reward(q):
#         global N, p
#         w = 0
#         for i in range(N):
#                 w += abs(q[i] - p[q[(i + 1) % N]])
#         return w

def reward(q):
        global N
        cnt = 0
        for i in range(N):
                for j in range(i):
                        if q[j] > q[i]:
                                cnt += 1
        return float(cnt)


#######################################################################################

model_DB = torch.nn.Sequential(
        OrderedDict([
                ("linear_1", torch.nn.Linear(N * N, hidden_size)),
                ("activation_1", torch.nn.ReLU()),
                ("linear_2", torch.nn.Linear(hidden_size, N * N + 1))
        ])
)
optimizer_DB = torch.optim.Adam(
        model_DB.parameters(),
        lr=lr
)


def loss_fn_DB():
        state = [-1] * N
        way = []
        while True:
                pred = model_DB(torch.tensor(transfer(state)))
                Pf = pred[:-1]
                for i in range(N):
                        if state[i] != -1:
                                for j in range(N):
                                        Pf[i * N + j] = -float("inf")
                                        Pf[j * N + state[i]] = -float("inf")
                Pf = torch.nn.functional.softmax(Pf, dim=0)
                F = pred[-1]
                ind = torch.distributions.categorical.Categorical(Pf).sample()
                Pb = 0
                for i in range(0, N):
                        if state[i] != -1:
                                Pb += 1
                state[ind // N] = ind % N
                if Pb == N - 1:
                        way.append([log(numpy.exp(-reward(state))), 0, 0])
                        break
                Pb = max(Pb, 1)
                way.append([F, torch.log(Pf[ind]), log(Pb)])
        loss = 0
        for i in range(0, len(way) - 1):
                loss += (way[i][0] + way[i][1] - way[i + 1][0] - way[i + 1][2]) ** 2
        return loss, state


#######################################################################################

model_TB_Pfs = torch.nn.Sequential(
        OrderedDict([
                ("linear_1", torch.nn.Linear(N * N, hidden_size)),
                ("activation_1", torch.nn.ReLU()),
                ("linear_2", torch.nn.Linear(hidden_size, N * N))
        ])
)
optimizer_TB_Pfs = torch.optim.Adam(
        model_TB_Pfs.parameters(),
        lr=lr
)

model_TB_log_Z = torch.nn.Parameter(torch.tensor(0.0, requires_grad=True))
optimizer_TB_log_Z = torch.optim.Adam([model_TB_log_Z], lr_Z)


def loss_fn_TB():
        state = [-1] * N
        way = []
        while True:
                Pf = model_TB_Pfs(torch.tensor(transfer(state)))
                for i in range(N):
                        if state[i] != -1:
                                for j in range(N):
                                        Pf[i * N + j] = -float("inf")
                                        Pf[j * N + state[i]] = -float("inf")
                Pf = torch.nn.functional.softmax(Pf, dim=0)
                ind = torch.distributions.categorical.Categorical(Pf).sample()
                Pb = 0
                for i in range(0, N):
                        if state[i] != -1:
                                Pb += 1
                state[ind // N] = ind % N
                if Pb == N - 1:
                        way.append([0, 0])
                        break
                Pb = max(Pb, 1)
                way.append([torch.log(Pf[ind]), log(Pb)])
        loss = model_TB_log_Z - log(numpy.exp(-reward(state)))
        for i in range(0, len(way) - 1):
                loss += way[i][0] - way[i + 1][1]
        loss = loss ** 2
        return loss, state


#######################################################################################

tmp = dict()
_ = 0
prop = []
for i in itertools.permutations(p):
        tmp[i] = _
        _ += 1
        prop.append(-reward(i))
prop = torch.nn.functional.softmax(torch.tensor(prop), dim=0)
rnd = torch.distributions.categorical.Categorical(prop)
cnt = [0] * _
for i in range(10000):
        cnt[rnd.sample()] += 1
for i in range(_):
        cnt[i] /= 10000

avg_cnt_DB = [0] * _
avg_cnt_TB = [0] * _


def empirical_loss_DB(samples=1000):
        global avg_cnt_DB
        loss = 0
        for i in range(_):
                avg_cnt_DB[i] /= samples
                loss += abs(avg_cnt_DB[i] - cnt[i])
        return loss


def empirical_loss_TB(samples=1000):
        global avg_cnt_DB
        loss = 0
        for i in range(_):
                avg_cnt_TB[i] /= samples
                loss += abs(avg_cnt_TB[i] - cnt[i])
        return loss


#######################################################################################

losses_DB = []
losses_TB = []
plt.ion()
for i in range(op):
        optimizer_DB.zero_grad()
        x = loss_fn_DB()
        loss = x[0]
        t = []
        for j in x[1]:
                t.append(j.item())
        avg_cnt_DB[tmp[tuple(t)]] += 1
        loss.backward()
        optimizer_DB.step()
        ###
        optimizer_TB_Pfs.zero_grad()
        optimizer_TB_log_Z.zero_grad()
        x = loss_fn_TB()
        loss = x[0]
        t = []
        for j in x[1]:
                t.append(j.item())
        avg_cnt_TB[tmp[tuple(t)]] += 1
        loss.backward()
        optimizer_TB_Pfs.step()
        optimizer_TB_log_Z.step()
        ###
        if i % 1000 == 0 and i != 0:
                print(i)
                x = empirical_loss_DB(1000)
                print(x)
                losses_DB.append(x)
                avg_cnt_DB = [0] * _
                x = empirical_loss_TB(1000)
                print(x)
                losses_TB.append(x)
                avg_cnt_TB = [0] * _
                clear_output(wait=True)
                plt.plot(losses_DB)
                plt.plot(losses_TB)
                plt.legend(['DB', 'TB'])
                plt.xlabel('iteraton')
                plt.ylabel('metric')
                plt.draw()
                plt.pause(0.001)
                plt.clf()
