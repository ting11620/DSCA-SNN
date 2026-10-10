import torch
import torch.nn as nn
from .dsca_lif import dsca_soma_update, readout_update

class ReadoutIntegrator(nn.Module):
    def __init__(self, input_dim, output_dim, device="cpu", bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = torch.device(device)
        self.dense = nn.Linear(input_dim, output_dim, bias=bias)
        self.tau_m = nn.Parameter(torch.empty(output_dim))
        nn.init.uniform_(self.tau_m, 0, 4)

    def set_neuron_state(self, batch_size):
        self.mem = torch.rand(batch_size, self.output_dim, device=self.device)

    def forward(self, input_spike):
        x = self.dense(input_spike.float())
        self.mem = readout_update(x, self.mem, self.tau_m)
        return self.mem

class DSCADense(nn.Module):
    def __init__(self, input_dim, output_dim, branch=4, device="cpu",
                 vth=1.0, bias=True, mask_share=1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.branch = branch
        self.device = torch.device(device)
        self.vth = vth
        self.mask_share = mask_share

        self.pad = ((input_dim // branch) * branch + branch - input_dim) % branch
        self.dense = nn.Linear(input_dim + self.pad, output_dim * branch, bias=bias)

        self.tau_m = nn.Parameter(torch.empty(output_dim))
        self.tau_n = nn.Parameter(torch.empty(output_dim, branch))
        self.ahp_gamm = nn.Parameter(torch.empty(output_dim))
        self.ahp_kapp = nn.Parameter(torch.empty(output_dim))
        self.feedback_factor = nn.Parameter(torch.empty(output_dim, branch))

        nn.init.uniform_(self.tau_m, 0, 4)
        nn.init.uniform_(self.tau_n, 2, 6)
        nn.init.uniform_(self.ahp_gamm, 2, 4)
        nn.init.uniform_(self.ahp_kapp, 0, 2)
        nn.init.uniform_(self.feedback_factor, -3, -1)

        self.register_buffer("mask", self._create_mask())

    def _create_mask(self):
        input_size = self.input_dim + self.pad
        mask = torch.zeros(self.output_dim * self.branch, input_size)
        for i in range(self.output_dim // self.mask_share):
            seq = torch.randperm(input_size)
            for j in range(self.branch):
                start = j * input_size // self.branch
                end = (j+1) * input_size // self.branch
                idx = seq[start:end]
                for k in range(self.mask_share):
                    row = (i*self.mask_share+k)*self.branch + j
                    mask[row, idx] = 1
        return mask

    def apply_mask(self):
        with torch.no_grad():
            self.dense.weight.mul_(self.mask)

    def set_neuron_state(self, batch_size):
        self.mem = torch.rand(batch_size, self.output_dim, device=self.device)
        self.spike = torch.rand(batch_size, self.output_dim, device=self.device)
        if self.branch == 1:
            self.d_input = torch.rand(batch_size, self.output_dim, self.branch, device=self.device)
        else:
            self.d_input = torch.zeros(batch_size, self.output_dim, self.branch, device=self.device)
        self.ahp_current = torch.zeros(batch_size, self.output_dim, device=self.device)
        self.v_th = torch.ones(batch_size, self.output_dim, device=self.device) * self.vth

    def forward(self, input_spike):
        beta = torch.sigmoid(self.tau_n)
        if self.pad:
            pad = torch.zeros(input_spike.size(0), self.pad, device=self.device)
            x = torch.cat((input_spike.float(), pad), dim=1)
        else:
            x = input_spike.float()

        coupling = -torch.sigmoid(self.feedback_factor)
        feedback = coupling.unsqueeze(0) * self.mem.unsqueeze(2)

        syn = self.dense(x).reshape(-1, self.output_dim, self.branch)
        self.d_input = beta * self.d_input + syn + feedback
        soma_input = self.d_input.sum(dim=2)

        self.mem, self.spike, self.ahp_current = dsca_soma_update(
            soma_input, self.mem, self.spike, self.v_th, self.tau_m,
            self.ahp_gamm, self.ahp_kapp, self.ahp_current
        )
        return self.mem, self.spike
