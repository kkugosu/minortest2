import torch
from torch import nn
from torch.nn.functional import normalize


def log_act(a):
    out = torch.where(a > 0, torch.log(a + 1), -torch.log(-a + 1))
    return out


class NValueNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(NValueNN, self).__init__()
        self.Linear_1 = nn.Linear(input_size, hidden_size)
        self.Linear_2 = nn.Linear(hidden_size, hidden_size)
        self.Linear_3 = nn.Linear(hidden_size, output_size)
        self.ELU = nn.ELU()

    def forward(self, input_element):
        output = self.Linear_1(input_element)
        output = log_act(output)
        output = self.Linear_2(output)
        output = log_act(output)
        output = self.Linear_3(output)
        output = normalize(output, dim=-1)
        return output


class ValueNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(ValueNN, self).__init__()
        self.Linear_1 = nn.Linear(input_size, hidden_size)
        self.Linear_2 = nn.Linear(hidden_size, hidden_size)
        self.Linear_3 = nn.Linear(hidden_size, output_size)
        self.ELU = nn.ELU()

    def forward(self, input_element):
        output = self.Linear_1(input_element)
        output = self.ELU(output)
        output = self.Linear_2(output)
        output = self.ELU(output)
        output = self.Linear_3(output)
        return output


class ProbNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(ProbNN, self).__init__()
        self.Linear_1 = nn.Linear(input_size, hidden_size)
        self.Linear_2 = nn.Linear(hidden_size, hidden_size)
        self.Linear_3 = nn.Linear(hidden_size, output_size)
        self.ELU = nn.ELU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_element):
        output = self.Linear_1(input_element)
        output = log_act(output)
        output = self.Linear_2(output)
        output = log_act(output)
        output = self.Linear_3(output)
        output = self.softmax(output)
        return output



