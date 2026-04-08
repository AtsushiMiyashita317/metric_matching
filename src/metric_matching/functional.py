import torch


class gated_log_impl(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, threshold: torch.Tensor, temperature: float):
        num = x.div(threshold).pow(temperature - 1).div(threshold)
        den = x.div(threshold).pow(temperature).add(1)
        y = torch.special.entr(x).neg().mul(num).div(den)
        ctx.save_for_backward(x, num, den, threshold)
        ctx.temperature = temperature
        return y
        
    @staticmethod
    def backward(ctx, grad_out):
        x, num, den, threshold = ctx.saved_tensors
        dx = num.div(den)
        num_x = x.div(threshold).pow(ctx.temperature - 2).div(threshold ** 2)
        dx = dx + torch.special.entr(x).neg().mul(ctx.temperature).mul(num_x).div(den.square())
        return grad_out * dx, None, None


def gated_log(x: torch.Tensor, threshold: torch.Tensor, temperature: float) -> torch.Tensor:
    return gated_log_impl.apply(x, threshold, temperature)

def gated_reciprocal(x: torch.Tensor, threshold: torch.Tensor, temperature: float) -> torch.Tensor:
    num = x.div(threshold).pow(temperature - 1).div(threshold)
    den = x.div(threshold).pow(temperature).add(1)
    return num.div(den)

def gated_ones(x: torch.Tensor, threshold: torch.Tensor, temperature: float) -> torch.Tensor:
    num = x.div(threshold).pow(temperature)
    den = x.div(threshold).pow(temperature).add(1)
    return num.div(den)
