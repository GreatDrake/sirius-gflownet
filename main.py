import torch

print(torch.__version__)
from math import log

D = 4
H = 8
R0 = 0.1
hidden_size = 100

from collections import OrderedDict

model = torch.nn.Sequential(
    OrderedDict([
        ("linear_1", torch.nn.Linear(D * H, hidden_size)),
        ("activation_1", torch.nn.ReLU()),
        ("linear_2", torch.nn.Linear(hidden_size, D + 2))
    ])
)

optimizer = torch.optim.SGD(
    model.parameters(),  # Передаём все тензоры, учавствующие в градиентном спуске
    lr=0.01,  # learning rate - величина шага градиентного спуска
)


def RIGHT_staaate(state):
    global H, D
    res = []
    for i in range(0, D):
        for j in range(0, H):
            res.append(float(0) + (state[i] == j))
    return res


def reward(x):
    global R0, D, H
    f1 = True
    f2 = True
    for d in range(D):
        if not (0.25 < abs(x[d] / (H - 1) - 0.5) <= 0.5): f1 = False
        if not (0.3 < abs(x[d] / (H - 1) - 0.5) < 0.4): f2 = False
    return (R0 + 0.5 * f1 + 2 * f2)


def loss_fn():
    global H, D
    state = [0] * D
    way = []
    while True:
        pred = model(torch.tensor(RIGHT_staaate(state)))
        p_Fs = pred[0:-1]
        F = pred[-1]
        p_B = 0
        for i in range(0, D):
            if state[i] == H - 1:
                p_Fs[i] = -float("inf")
            if state[i] > 0:
                p_B += 1
        m = torch.nn.Softmax(dim=0)
        p_Fs = m(p_Fs)
        ind = torch.distributions.categorical.Categorical(p_Fs).sample()
        p_B = max(p_B, 1)
        way.append([F, torch.log(p_Fs[ind]), 1 / p_B])
        if ind == D:
            break
        state[ind] += 1
    loss = 0
    for i in range(0, len(way) - 1):
        loss += (way[i][0] + way[i][1] - way[i + 1][0] - log(way[i + 1][2])) ** 2
    loss += (way[-1][0] + way[-1][1] - reward(state) - log(1)) ** 2
    return loss


def random_WAy():
    global H, D
    state = [0] * D
    while True:
        pred = model(torch.tensor(RIGHT_staaate(state)))
        p_Fs = pred[0:-1]
        for i in range(0, D):
            if state[i] == H - 1:
                p_Fs[i] = -float("inf")
        m = torch.nn.Softmax(dim=0)
        p_Fs = m(p_Fs)
        ind = torch.distributions.categorical.Categorical(p_Fs).sample()
        if ind == D:
            break
        state[ind] += 1
    return state


op = 10000
cnt = 10

for i in range(0, op):
    optimizer.zero_grad()
    loss = loss_fn()
    print(loss * 100)
    loss.backward()
    optimizer.step()
