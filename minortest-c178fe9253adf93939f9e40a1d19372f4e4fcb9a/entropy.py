import torch
import basic_nn
import narrow
import numpy as np
from torch import nn
from torch.nn.functional import normalize
import copy
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.autograd.set_detect_anomaly(True)

env = narrow.Narrow()
upd_policy = basic_nn.ValueNN(2, 256, 1).to(DEVICE)
upd_queue = basic_nn.ValueNN(3, 256, 1).to(DEVICE)
base_queue = basic_nn.ValueNN(3, 256, 1).to(DEVICE)
skill_num = 4
traj_len = 3
l_r = 1e-4
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
    _mat = _mat + torch.eye(skill_num * traj_len).to(DEVICE)
    _mat = torch.where(_mat > 0, _mat, 0.000001)
    _mat = torch.log(_mat)
    return torch.sum(_mat)


def cal_reward_2(ary_1, ary_2, i):
    ary_1 = ary_1.squeeze()
    ary_2 = ary_2.squeeze()

    traj_idx = int(i % traj_len)
    skill_idx = int((i - traj_idx)/skill_num)

    a = torch.square(ary_1[i] - ary_1)
    a = a + torch.eye(len(ary_1)).to(DEVICE)
    a = torch.where(a > 0, a, 0.000001)
    a = torch.log(a)

    b = torch.square(ary_1[i] - ary_1[i: (skill_idx+1) * traj_len])
    b = b + torch.eye(len(ary_1[i: (skill_idx+1) * traj_len])).to(DEVICE)
    b = torch.where(b > 0, b, 0.000001)
    b = torch.log(b)

    ary1_value = torch.sum(a) - torch.sum(b)

    a = torch.square(ary_2[i] - ary_2)
    a = a + torch.eye(len(ary_2)).to(DEVICE)
    a = torch.where(a > 0, a, 0.000001)
    a = torch.log(a)

    b = torch.square(ary_2[i] - ary_2[i: (skill_idx + 1) * traj_len])
    b = b + torch.eye(len(ary_1[i: (skill_idx+1) * traj_len])).to(DEVICE)
    b = torch.where(b > 0, b, 0.000001)
    b = torch.log(b)

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
            lr_q.append(l_r)
        lr_q.append(l_r)
        weight_decay_q.append(0.1)
    upd_queue_list.append(tmp_queue)

    tmp_base_queue = copy.deepcopy(base_queue)
    base_queue_list.append(tmp_base_queue)

    policy_list.append(tmp_policy)

    i = i + 1

optimizer_p = torch.optim.SGD([{'params': p, 'lr': l, 'weight_decay': d} for p, l, d in zip(network_p, lr_p, weight_decay_p)])
optimizer_q = torch.optim.SGD([{'params': p, 'lr': l, 'weight_decay': d} for p, l, d in zip(network_q, lr_q, weight_decay_q)])
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
            n_action = t_action.cpu().detach().numpy()/10
            n_state, _, _ = env.step(n_action[0])
            t_state = torch.from_numpy(n_state).to(DEVICE).type(torch.float32)
            l = l + 1
        k = k + 1
    i = 0
    while i < skill_num:
        if i == 0:
            net_action = policy_list[i](input_list[i])/10
        else:
            net_action = torch.cat((net_action, policy_list[i](input_list[i])/10), 0)
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

    another_out = input_list[:, :, -1].reshape(-1, 1).repeat(1, len(index)).reshape(1, -1).repeat(traj_len * skill_num, 1)
    another_out2 = input_list.reshape(-1, 2).repeat(1, len(index), 1).reshape(1, -1, 2).repeat(traj_len * skill_num, 1, 1)
    another_out = torch.clamp(another_out + another_action, min=-1, max=1)

    queue_input = torch.cat((another_out2, another_action.unsqueeze(-1)), -1)
    queue_input = queue_input.reshape(traj_len * skill_num, skill_num * traj_len, len(index), 3)

    reward = torch.zeros(traj_len * skill_num * len(index)).to(DEVICE)
    _another_out = another_out.reshape(traj_len * skill_num, traj_len * skill_num, len(index))
    i = 0
    loss2 = 0
    while i < skill_num:
        j = 0
        while j < traj_len:
            k = 0
            while k < len(index):

                a = upd_queue_list[i](torch.transpose(queue_input[i * traj_len + j], 0, 1)[k][i * traj_len + j])
                with torch.no_grad():
                    b = cal_reward_2(_another_out[i * traj_len + j].T[k], net_out, i * traj_len + j)*1e-4

                loss2 = loss2 + criterion(a[0], b)
                k = k + 1
            j = j + 1
        i = i + 1
    print(loss2)
    """
    full_idx = 0
    while full_idx < traj_len*skill_num*len(index):
        index_1 = int(full_idx % (len(index)))
        index_2 = int((full_idx - index_1)/(len(index)))
        # cal_reward_2(_another_out[index_2].T[index_1], net_out, index_2)

        # cal_reward(_another_out[index_2].T[index_1])
        full_idx = full_idx + 1
    """

    final_reward = reward.reshape(traj_len * skill_num, len(index))*1e-4
    i = 0

    while i < skill_num:
        j = 0
        while j < traj_len:
            k = 0
            while k < len(index):
                with torch.no_grad():
                    # r = upd_queue_list[i](torch.transpose(queue_input[i * traj_len + j], 0, 1)[k][i * traj_len + j])
                    r = cal_reward_2(_another_out[i * traj_len + j].T[k], net_out, i * traj_len + j)*1e-4
                final_reward[i * traj_len + j, k] = r
                k = k + 1
            j = j + 1
        i = i + 1

    prob_matrix = index.repeat(traj_len * skill_num, 1)
    prob = (-1/2)*torch.square(net_action.reshape(-1, 1).repeat(1, len(index)) - prob_matrix)
    loss = torch.sum(torch.exp(final_reward)*(final_reward-prob))# + torch.sum(torch.exp(prob)*(prob - final_reward))

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
    loss2.backward(retain_graph=True)
    i = 0  # seq training
    while i < len(upd_queue_list):
        for param in upd_queue_list[i].parameters():
            param.register_hook(lambda grad: torch.nan_to_num(grad, nan=0.0))
            param.grad.data.clamp_(-1, 1)
        i = i + 1
    optimizer_q.step()
    i = 0

    iter = iter + 1

