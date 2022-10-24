import torch
import basic_nn
import narrow
import numpy as np
from torch import nn
from torch.nn.functional import normalize
import copy
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.autograd.set_detect_anomaly(True)
GAMMA = 0.90

env = narrow.Narrow()
upd_policy = basic_nn.ValueNN(2, 256, 1).to(DEVICE)
upd_queue = basic_nn.ValueNN(3, 256, 1).to(DEVICE)
base_queue = basic_nn.ValueNN(3, 256, 1).to(DEVICE)
skill_num = 10
traj_len = 11
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
    # print("calrew", ary)
    return torch.sum(torch.square(ary.unsqueeze(-1).T - ary.unsqueeze(-1)))


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
    policy_list.append(tmp_policy)

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

    i = i + 1
print("assertion")

optimizer_p = torch.optim.SGD([{'params': p, 'lr': l, 'weight_decay': d} for p, l, d in zip(network_p, lr_p, weight_decay_p)])
optimizer_q = torch.optim.SGD([{'params': p, 'lr': l, 'weight_decay': d} for p, l, d in zip(network_q, lr_q, weight_decay_q)])
criterion = nn.MSELoss(reduction='mean')

net_action = torch.randn((1, skill_num), requires_grad=True).type(torch.float32).to(DEVICE)
j = 0
input_list = torch.zeros((skill_num, traj_len, 2)).to(DEVICE)

while j < 1000:
    while i < skill_num:
        base_queue_list[i].load_state_dict(upd_queue_list[i].state_dict())
        base_queue_list[i].eval()
    k = 0
    while k < skill_num:
        n_state = env.reset()
        t_state = torch.from_numpy(n_state).to(DEVICE).type(torch.float32)
        l = 0
        while l < traj_len:
            t_action = policy_list[k](t_state)
            t_action = torch.clamp(t_action/10, min=-0.1, max=0.1)
            input_list[k][l] = t_state
            n_action = t_action.cpu().detach().numpy()
            n_state, _, _ = env.step(n_action[0])
            t_state = torch.from_numpy(n_state).to(DEVICE).type(torch.float32)
            l = l + 1
        k = k + 1
    print(j)
    i = 0
    while i < skill_num:
        if i == 0:
            net_action = policy_list[i](input_list[i])/10
        else:
            net_action = torch.cat((net_action, policy_list[i](input_list[i])/10), 0)
        i = i + 1
    net_action = torch.clamp(net_action, min=-0.1, max=0.1)
    # net action size 110, 1
    index = (torch.arange(21).to(DEVICE) - 10) / 10
    with torch.no_grad():
        another_action = net_action.reshape(-1, 1).repeat(1, len(index)).reshape(1, -1).repeat(traj_len * skill_num, 1)
    idx = 0
    while idx < traj_len * skill_num:
        another_action[idx][len(index) * idx: len(index) * (idx + 1)] = index
        idx = idx + 1
    another_action = torch.clamp(another_action, min=-0.5, max=0.5)

    net_out = torch.clamp(input_list[:, :, -1].reshape(-1, 1) + net_action, min=-1, max=1)
    mat = torch.square(net_out.T - net_out)

    base_reward = torch.sum(mat)

    another_out = input_list[:, :, -1].reshape(-1, 1).repeat(1, len(index)).reshape(1, -1).repeat(traj_len * skill_num, 1)
    another_out = torch.clamp(another_out + another_action, min=-1, max=1)

    reward = torch.zeros(traj_len * skill_num * len(index)).to(DEVICE)
    _another_out = another_out.reshape(traj_len * skill_num, traj_len * skill_num, len(index))
    full_idx = 0
    while full_idx < traj_len*skill_num*len(index):
        index_1 = int(full_idx % (len(index)))
        index_2 = int((full_idx - index_1)/(len(index)))
        reward[full_idx] = cal_reward(_another_out[index_2].T[index_1])
        full_idx = full_idx + 1
    final_reward = (reward - base_reward).reshape(traj_len * skill_num, len(index))

    queue_base_reward = torch.zeros(traj_len * skill_num).to(DEVICE)
    tmp_action = net_action.squeeze().unsqueeze(0).repeat(traj_len * skill_num, 1)

    i = 0
    while i < traj_len * skill_num:
        tmp_action[i][i] = 0
        queue_base_reward[i] = cal_reward(input_list[:, :, -1].reshape(-1) + tmp_action[i])
        #print("compare")
        #print(base_reward)
        #print(queue_base_reward[i])
        i = i + 1
    queue_base_reward = queue_base_reward.reshape(skill_num, traj_len)

    skill_id = 0
    queue_loss = 0
    with torch.no_grad():
        queue_input = torch.cat((input_list, net_action.reshape(skill_num, traj_len, 1)), -1)
    trans_action = torch.zeros((skill_num, traj_len, 2)).to(DEVICE)
    trans_action[:, :, -1] = net_action.reshape(skill_num, traj_len)
    next_state = input_list + trans_action
    #print("base1", base_reward)
    #print("base2", queue_base_reward)
    queue_reward = base_reward - queue_base_reward
    # print(queue_reward)
    while skill_id < skill_num:
        t_p_qvalue = upd_queue_list[skill_id](queue_input[skill_id]).squeeze()
        next_action = policy_list[skill_id](next_state[skill_id])
        base_queue_input = torch.cat((next_state[skill_id], next_action), -1)
        with torch.no_grad():
            t_qvalue = queue_reward[skill_id]/10 + GAMMA * base_queue_list[skill_id](base_queue_input).squeeze()
        print(t_p_qvalue)
        print(queue_reward[skill_id])
        print(base_queue_list[skill_id](base_queue_input).squeeze())
        queue_loss = queue_loss + criterion(t_p_qvalue, t_qvalue)
        skill_id = skill_id + 1
    print("queueloss")
    print(queue_loss)
    prob_matrix = index.repeat(traj_len * skill_num, 1)
    prob = (-1/2)*torch.square(net_action.reshape(-1, 1).repeat(1, len(index)) - prob_matrix)

    loss = -torch.sum(torch.exp(final_reward/100)*prob)
    # another out and another act input?? (traj*skill, len(index))
    final_r_input = torch.cat((prob_matrix.unsqueeze(-1), input_list.reshape(-1, 1, 2).repeat(1, len(index), 1)), -1)
    final_reward = torch.zeros((skill_num, traj_len * len(index))).to(DEVICE)
    i = 0
    while i < skill_num:
        final_reward[i] = base_queue_list[i](
            final_r_input.reshape(skill_num, -1, 3)[i]).squeeze() #something wrong..
        i = i + 1
    with torch.no_grad():
        final_reward = final_reward.reshape(skill_num*traj_len, len(index))
    loss = -torch.sum(torch.exp(final_reward / 100) * prob)
    # print(reward[:100])
    print("final_reward")
    print(base_reward)
    print(reward[:100])
    print(another_action[:10])
    print(net_action.squeeze()[:10])
    print(final_reward[:10])
    print(torch.sort(net_out.squeeze())[0])
    optimizer_p.zero_grad()
    loss.backward(retain_graph=True)
    i = 0  # seq training
    while i < len(policy_list):
        for param in policy_list[i].parameters():
            # print(param)
            param.register_hook(lambda grad: torch.nan_to_num(grad, nan=0.0))
            param.grad.data.clamp_(-1, 1)
        i = i + 1
    optimizer_p.step()

    optimizer_q.zero_grad()
    queue_loss.backward(retain_graph=True)
    i = 0  # seq training
    while i < len(upd_queue_list):
        for param in upd_queue_list[i].parameters():
            # print(param)
            param.register_hook(lambda grad: torch.nan_to_num(grad, nan=0.0))
            param.grad.data.clamp_(-1, 1)
        i = i + 1
    optimizer_q.step()
    j = j + 1

