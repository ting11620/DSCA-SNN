import math
import torch
import torch.nn.functional as F

SURROGATE_TYPE = "MG"
SURROGATE_GAMMA = 0.5
SURROGATE_LENS = 0.5
R_M = 1.0

def gaussian(x, mu=0.0, sigma=0.5):
    pi = torch.tensor(math.pi, dtype=x.dtype, device=x.device)
    return torch.exp(-((x-mu)**2)/(2*sigma**2)) / torch.sqrt(2*pi) / sigma

class ActFunAdp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.gt(0).float()

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        scale, height = 6.0, 0.15
        if SURROGATE_TYPE == "MG":
            temp = (
                gaussian(x, 0.0, SURROGATE_LENS) * (1.0 + height)
                - gaussian(x, SURROGATE_LENS, scale*SURROGATE_LENS) * height
                - gaussian(x, -SURROGATE_LENS, scale*SURROGATE_LENS) * height
            )
        elif SURROGATE_TYPE == "G":
            pi = torch.tensor(math.pi, dtype=x.dtype, device=x.device)
            temp = torch.exp(-(x**2)/(2*SURROGATE_LENS**2)) / torch.sqrt(2*pi) / SURROGATE_LENS
        elif SURROGATE_TYPE == "linear":
            temp = F.relu(1-x.abs())
        elif SURROGATE_TYPE == "slayer":
            temp = torch.exp(-5*x.abs())
        elif SURROGATE_TYPE == "rect":
            temp = x.abs() < 0.5
        else:
            raise ValueError(SURROGATE_TYPE)
        return grad_output.clone() * temp.float() * SURROGATE_GAMMA

spike_fn = ActFunAdp.apply

def dsca_soma_update(inputs, mem, spike, v_th, tau_m, ahp_gamm, ahp_kapp, ahp_current):
    alpha = torch.sigmoid(tau_m)
    ahp_gamma = torch.sigmoid(ahp_gamm)
    ahp_kappa = torch.sigmoid(ahp_kapp)
    ahp_current = ahp_gamma * ahp_current + ahp_kappa * spike
    mem = mem * alpha * (1-spike) + (1-alpha) * (R_M*inputs - ahp_current)
    spike = spike_fn(mem - v_th)
    return mem, spike, ahp_current

def readout_update(inputs, mem, tau_m):
    alpha = torch.sigmoid(tau_m)
    return mem * alpha + (1-alpha) * inputs
