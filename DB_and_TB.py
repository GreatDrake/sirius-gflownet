import torch
from math import log
from collections import OrderedDict
from itertools import product

#######################################################################################

D = 4
H = 8
R0 = 0.1
op = 100000
batch = 10
hidden_size = 100
lr = 0.0001
lr_Z = 0.01


#######################################################################################


def transfer(state):
    global H, D, batch
    res = [[float(0)] * H * D for i in range(batch)]
    for k in range(0, batch):
        for i in range(0, D):
            res[k][state[k][i] + i * H] = float(1)
    return res


def reward(x):
    global R0, D, H
    f1 = True
    f2 = True
    for d in range(D):
        if not (0.25 < abs(x[d] / (H - 1) - 0.5) <= 0.5): f1 = False
        if not (0.3 < abs(x[d] / (H - 1) - 0.5) < 0.4): f2 = False
    return R0 + 0.5 * f1 + 2 * f2


#######################################################################################

model_DB = torch.nn.Sequential(
    OrderedDict([
        ("linear_1", torch.nn.Linear(D * H, hidden_size)),
        ("activation_1", torch.nn.ReLU()),
        ("linear_2", torch.nn.Linear(hidden_size, D + 2))
    ])
)
optimizer_DB = torch.optim.Adam(
    model_DB.parameters(),
    lr=lr
)


def loss_fn_DB():
    global H, D, batch
    used = [0] * batch
    states = [[0] * D for i in range(batch)]
    ways = [[] for i in range(batch)]
    while True:
        pred = model_DB(torch.tensor(transfer(states)))
        p_Fs = pred[:, 0: -1]
        Fs = pred[:, -1]
        end = True
        for j in range(0, batch):
            if used[j]:
                continue
            end = False
            for i in range(0, D):
                if states[j][i] == H - 1:
                    p_Fs[j][i] = -float("inf")
        if end:
            break
        m = torch.nn.Softmax(dim=1)
        p_Fs = m(p_Fs)
        while True:
            ind = torch.distributions.categorical.Categorical(p_Fs).sample()
            iscorrect = True
            for j in range(0, batch):
                if used[j] or ind[j] == D:
                    continue
                if states[j][ind[j]] == H - 1:
                    iscorrect = False
            if iscorrect == False:
                continue
            for j in range(0, batch):
                if used[j]:
                    continue
                p_B = 0
                for i in range(0, D):
                    if states[j][i] >= 1:
                        p_B += 1
                p_B = max(p_B, 1)
                ways[j].append([Fs[j], torch.log(p_Fs[j][ind[j]]), log(1 / p_B)])
                if ind[j] == D:
                    ways[j].append([log(reward(states[j])), 0, 0])
                    used[j] = True
                else:
                    assert (states[j][ind[j]] != H - 1)
                    states[j][ind[j]] += 1
            break
    loss = 0
    for j in range(batch):
        for i in range(0, len(ways[j]) - 1):
            loss += (ways[j][i][0] + ways[j][i][1] - ways[j][i + 1][0] - ways[j][i + 1][2]) ** 2
    loss /= batch
    return loss, states


#######################################################################################

model_TB_Pfs = torch.nn.Sequential(
    OrderedDict([
        ("linear_1", torch.nn.Linear(D * H, hidden_size)),
        ("activation_1", torch.nn.ReLU()),
        ("linear_2", torch.nn.Linear(hidden_size, D + 1))
    ])
)
optimizer_TB_Pfs = torch.optim.Adam(
    model_TB_Pfs.parameters(),
    lr=lr
)

model_TB_log_Z = torch.nn.Parameter(torch.tensor(0.0, requires_grad=True))
optimizer_TB_log_Z = torch.optim.Adam([model_TB_log_Z], lr_Z)


def loss_fn_TB():
    global H, D, model_TB_log_Z, batch
    used = [0] * batch
    allstate = [[0] * D for i in range(batch)]
    way = [[] for i in range(batch)]
    while True:
        p_Fs = model_TB_Pfs(torch.tensor(transfer(allstate)))
        was = False
        for j in range(0, batch):
            if used[j]:
                continue
            was = True
            for i in range(0, D):
                if allstate[j][i] == H - 1:
                    p_Fs[j][i] = -float("inf")
        if was == False:
            break
        m = torch.nn.Softmax(dim=1)
        p_Fs = m(p_Fs)
        while True:
            ind = torch.distributions.categorical.Categorical(p_Fs).sample()
            iscorrect = True
            for j in range(0, batch):
                if used[j] or ind[j] == D:
                    continue
                if allstate[j][ind[j]] == H - 1:
                    iscorrect = False
            if iscorrect == False:
                continue
            for j in range(0, batch):
                if used[j]:
                    continue
                p_B = 0
                for i in range(0, D):
                    if allstate[j][i] >= 1:
                        p_B += 1
                p_B = max(p_B, 1)
                way[j].append([torch.log(p_Fs[j][ind[j]]), log(1 / p_B)])
                if ind[j] == D:
                    #way[j][-1][1] = 1
                    way[j].append([0, 0])
                    used[j] = True
                else:
                    assert(allstate[j][ind[j]] != H - 1)
                    allstate[j][ind[j]] += 1
            break
    j = 0
    loss = 0
    for state in allstate:
        cur = model_TB_log_Z - log(reward(state))
        for i in range(0, len(way[j]) - 1):
            cur += way[j][i][0] - way[j][i + 1][1]
        cur = cur ** 2
        loss += cur
        j += 1
    loss /= batch
    return loss, allstate


#######################################################################################

rewards = torch.zeros(*[H for i in range(D)])
coord_diap = [range(H) for _ in range(D)]

for coord in product(*coord_diap):
    rewards[tuple(coord)] = reward(torch.tensor(coord))

rewards /= rewards.sum()


def empirical_loss(a, samples=1000):
    counter = torch.zeros(*[H for i in range(D)])
    for i in range(samples):
        counter[tuple(a[i])] += 1

    counter /= counter.sum()

    return (rewards - counter).abs().sum()


#######################################################################################

visited_DB = []
visited_TB = []
for i in range(op):
    optimizer_DB.zero_grad()
    x = loss_fn_DB()
    loss_DB = x[0]
    for j in x[1]:
        visited_DB.append(j)
    loss_DB.backward()
    optimizer_DB.step()
    ####
    optimizer_TB_Pfs.zero_grad()
    optimizer_TB_log_Z.zero_grad()
    x = loss_fn_TB()
    loss_TB = x[0]
    for j in x[1]:
        visited_TB.append(j)
    loss_TB.backward()
    optimizer_TB_Pfs.step()
    optimizer_TB_log_Z.step()
    ###
    if i % 100 == 0:
        print(i * batch)
        print(end="DB: ")
        print(empirical_loss(visited_DB[-10000:], len(visited_DB[-10000:])))
        print(end="TB: ")
        print(empirical_loss(visited_TB[-10000:], len(visited_TB[-10000:])))
