import torch
import basic_nn
import narrow
import numpy as np
from torch import nn
from torch.nn.functional import normalize
import copy

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.autograd.set_detect_anomaly(True)
GAMMA = 0.98

env = narrow.Narrow()
upd_policy = basic_nn.ValueNN(2, 256, 1).to(DEVICE)
upd_queue = basic_nn.ValueNN(3, 256, 1).to(DEVICE)
base_queue = basic_nn.ValueNN(3, 256, 1).to(DEVICE)
skill_num = 10
traj_len = 11
l_r = 1e-5
# entropy is nore faster and flucturate, kld is less flucturate and slower
policy_list = []
network_p = []
lr_p = []
weight_decay_p = []
network_q = []
lr_q = []
weight_decay_q = []
upd_queue_list = []
base_queue_list = []


def cal_reward(ary):
    ary = ary.squeeze()
    _mat = torch.square(ary.unsqueeze(-1).T - ary.unsqueeze(-1))
    return torch.sum(_mat)


def cal_reward_2(ary_1, ary_2, i):
    ary_1 = ary_1.squeeze()
    ary_2 = ary_2.squeeze()

    traj_idx = int(i % traj_len)
    skill_idx = int((i - traj_idx) / skill_num)

    a = torch.square(ary_1[i] - ary_1)

    b = torch.square(ary_1[i] - ary_1[i: (skill_idx + 1) * traj_len])

    ary1_value = torch.sum(a) - torch.sum(b)

    a = torch.square(ary_2[i] - ary_2)

    b = torch.square(ary_2[i] - ary_2[i: (skill_idx + 1) * traj_len])

    ary2_value = torch.sum(a) - torch.sum(b)
    return ary1_value - ary2_value


i = 0
while i < skill_num:

    tmp_policy = copy.deepcopy(upd_policy)

    assert tmp_policy is not upd_policy, "copy error"
    for name, param in tmp_policy.named_parameters():

        torch.nn.init.uniform_(param, -0.1, 0.1)
        param.register_hook(lambda grad: torch.nan_to_num(grad, nan=0.0))
        network_p.append(param)
        if name == "Linear_1.bias":
            lr_p.append(l_r * 10)
        else:
            lr_p.append(l_r)
        weight_decay_p.append(0.1)

    tmp_queue = copy.deepcopy(upd_queue)
    assert tmp_queue is not upd_queue, "copy error"

    for name, param in tmp_queue.named_parameters():
        torch.nn.init.uniform_(param, -0.2, 0.2)
        param.register_hook(lambda grad: torch.nan_to_num(grad, nan=0.0))
        network_q.append(param)
        if name == "Linear_1.bias":
            lr_q.append(l_r * 10)
        else:
            lr_q.append(l_r * 1)
        lr_q.append(l_r)
        weight_decay_q.append(0.1)
    upd_queue_list.append(tmp_queue)

    tmp_base_queue = copy.deepcopy(base_queue)
    base_queue_list.append(tmp_base_queue)

    policy_list.append(tmp_policy)

    i = i + 1

optimizer_p = torch.optim.SGD(
    [{'params': p, 'lr': l, 'weight_decay': d} for p, l, d in zip(network_p, lr_p, weight_decay_p)])
optimizer_q = torch.optim.SGD(
    [{'params': p, 'lr': l, 'weight_decay': d} for p, l, d in zip(network_q, lr_q, weight_decay_q)])
criterion = nn.MSELoss(reduction='mean')

net_action = torch.randn((1, skill_num), requires_grad=True).type(torch.float32).to(DEVICE)
iter = 0
input_list = torch.zeros((skill_num, traj_len, 2)).to(DEVICE)
while iter < 1000:
    print(iter)
    k = 0
    while k < skill_num:
        n_state = env.reset()
        t_state = torch.from_numpy(n_state).to(DEVICE).type(torch.float32)
        l = 0
        while l < traj_len:
            t_action = policy_list[k](t_state)
            input_list[k][l] = t_state
            n_action = t_action.cpu().detach().numpy() / 10
            n_state, _, _ = env.step(n_action[0])
            t_state = torch.from_numpy(n_state).to(DEVICE).type(torch.float32)
            l = l + 1
        k = k + 1
    i = 0
    while i < skill_num:
        if i == 0:
            net_action = policy_list[i](input_list[i]) / 10
        else:
            net_action = torch.cat((net_action, policy_list[i](input_list[i]) / 10), 0)
        i = i + 1
    # net action size 110, 1
    index = (torch.arange(21).to(DEVICE) - 10) / 100
    with torch.no_grad():
        another_action = net_action.reshape(-1, 1).repeat(1, len(index)).reshape(1, -1).repeat(traj_len * skill_num, 1)
    idx = 0
    while idx < traj_len * skill_num:
        another_action[idx][len(index) * idx: len(index) * (idx + 1)] = index
        idx = idx + 1

    net_out = torch.clamp(input_list[:, :, -1].reshape(-1, 1) + net_action, min=-1, max=1)

    another_out = input_list[:, :, -1].reshape(-1, 1).repeat(1, len(index)).reshape(1, -1).repeat(traj_len * skill_num,
                                                                                                  1)
    another_out2 = input_list.reshape(-1, 2).repeat(1, len(index), 1).reshape(1, -1, 2).repeat(traj_len * skill_num, 1,
                                                                                               1)
    another_out = torch.clamp(another_out + another_action, min=-1, max=1)

    reward = torch.zeros(traj_len * skill_num * len(index)).to(DEVICE)
    final_reward = reward.reshape(traj_len * skill_num, len(index)) * 1e-4

    _another_out = another_out.reshape(traj_len * skill_num, traj_len * skill_num, len(index))

    trans_action = torch.ones((skill_num, traj_len, 2)).to(DEVICE)
    trans_action[:, :, -1] = net_action.reshape(skill_num, traj_len)
    next_state = input_list + trans_action

    queue_reward = torch.zeros(traj_len * skill_num).to(DEVICE)
    tmp_action = net_action.squeeze().unsqueeze(0).repeat(traj_len * skill_num, 1)

    i = 0
    while i < traj_len * skill_num:
        tmp_action[i][i] = 0
        queue_reward[i] = cal_reward_2(net_out, input_list[:, :, -1].reshape(-1) + tmp_action[i], i)  # 0 to ith index?
        i = i + 1
    queue_reward = queue_reward.reshape(skill_num, traj_len)
    with torch.no_grad():
        queue_input = torch.cat((input_list, net_action.reshape(skill_num, traj_len, 1)), -1)
    skill_id = 0
    queue_loss = 0
    while skill_id < skill_num:
        t_p_qvalue = upd_queue_list[skill_id](queue_input[skill_id]).squeeze()
        next_action = policy_list[skill_id](next_state[skill_id]) / 10
        base_queue_input = torch.cat((next_state[skill_id], next_action), -1)
        with torch.no_grad():
            t_qvalue = queue_reward[skill_id] * 1e-4  # + GAMMA * base_queue_list[skill_id](base_queue_input).squeeze()
        queue_loss = queue_loss + criterion(t_p_qvalue, t_qvalue)
        skill_id = skill_id + 1
    print(queue_loss)

    queue_input = torch.cat((another_out2, another_action.unsqueeze(-1)), -1)
    queue_input = queue_input.reshape(traj_len * skill_num, skill_num * traj_len, len(index), 3)
    sk_idx = 0
    while sk_idx < skill_num:
        tr_idx = 0
        while tr_idx < traj_len:
            idx = 0
            while idx < len(index):
                index_1 = idx
                index_2 = sk_idx * traj_len + tr_idx
                with torch.no_grad():
                    final_reward[index_2][index_1] = \
                        upd_queue_list[sk_idx](torch.transpose(queue_input[index_2], 0, 1)[index_1][index_2])
                idx = idx + 1
            tr_idx = tr_idx + 1
        sk_idx = sk_idx + 1

    prob_matrix = index.repeat(traj_len * skill_num, 1)
    prob = (-1 / 2) * torch.square(net_action.reshape(-1, 1).repeat(1, len(index)) - prob_matrix)
    loss = torch.sum(torch.exp(prob) * (prob - final_reward))
    # loss = torch.sum(torch.exp(final_reward)*(final_reward-prob))
    # loss = torch.sum(torch.exp(final_reward)*(final_reward-prob)) + torch.sum(torch.exp(prob)*(prob - final_reward))
    print(torch.sort(net_out.squeeze())[0])
    optimizer_p.zero_grad()
    loss.backward(retain_graph=True)
    i = 0  # seq training
    while i < len(policy_list):
        for param in policy_list[i].parameters():
            param.register_hook(lambda grad: torch.nan_to_num(grad, nan=0.0))
            param.grad.data.clamp_(-1, 1)
        i = i + 1
    optimizer_p.step()

    optimizer_q.zero_grad()
    queue_loss.backward(retain_graph=True)
    i = 0  # seq training
    while i < len(upd_queue_list):
        for param in upd_queue_list[i].parameters():
            param.register_hook(lambda grad: torch.nan_to_num(grad, nan=0.0))
            param.grad.data.clamp_(-1, 1)
        i = i + 1
    optimizer_q.step()
    i = 0

    iter = iter + 1

