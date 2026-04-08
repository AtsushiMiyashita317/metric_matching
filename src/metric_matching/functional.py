import torch


class gated_log_impl(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, threshold: torch.Tensor, temperature: float):
        y = torch.empty_like(x)
        left = x <= threshold
        right = ~left

        if left.any():
            x_left = x[left]
            threshold_left = threshold[left]
            ratio = x_left.div(threshold_left)
            num = ratio.pow(temperature - 1).div(threshold_left)
            den = ratio.pow(temperature).add(1)
            y[left] = torch.special.entr(x_left).neg().mul(num).div(den)

        if right.any():
            x_right = x[right]
            threshold_right = threshold[right]
            inv_ratio = threshold_right.div(x_right)
            den = inv_ratio.pow(temperature).add(1)
            y[right] = x_right.log().div(den)

        ctx.save_for_backward(x, threshold)
        ctx.temperature = temperature
        return y

    @staticmethod
    def backward(ctx, grad_out):
        x, threshold = ctx.saved_tensors
        dx = torch.empty_like(x)
        left = x <= threshold
        right = ~left

        if left.any():
            x_left = x[left]
            threshold_left = threshold[left]
            ratio = x_left.div(threshold_left)
            num = ratio.pow(ctx.temperature - 1).div(threshold_left)
            den = ratio.pow(ctx.temperature).add(1)
            num_x = ratio.pow(ctx.temperature - 2).div(threshold_left.square())
            dx[left] = num.div(den) + (
                torch.special.entr(x_left)
                .neg()
                .mul(ctx.temperature)
                .mul(num_x)
                .div(den.square())
            )

        if right.any():
            x_right = x[right]
            threshold_right = threshold[right]
            inv_ratio = threshold_right.div(x_right)
            u = inv_ratio.pow(ctx.temperature)
            den = u.add(1)
            dx[right] = x_right.reciprocal().div(den) + (
                x_right.log()
                .mul(ctx.temperature)
                .mul(u)
                .div(x_right)
                .div(den.square())
            )

        return grad_out * dx, None, None


def gated_log(x: torch.Tensor, threshold: torch.Tensor, temperature: float) -> torch.Tensor:
    threshold = threshold.expand_as(x)
    return gated_log_impl.apply(x, threshold, temperature)

def gated_reciprocal(x: torch.Tensor, threshold: torch.Tensor, temperature: float) -> torch.Tensor:
    threshold = threshold.expand_as(x)
    y = torch.empty_like(x)
    left = x <= threshold
    right = ~left

    if left.any():
        x_left = x[left]
        threshold_left = threshold[left]
        ratio = x_left.div(threshold_left)
        num = ratio.pow(temperature - 1).div(threshold_left)
        den = ratio.pow(temperature).add(1)
        y[left] = num.div(den)

    if right.any():
        x_right = x[right]
        threshold_right = threshold[right]
        inv_ratio = threshold_right.div(x_right)
        den = inv_ratio.pow(temperature).add(1)
        y[right] = x_right.reciprocal().div(den)

    return y

def gated_ones(x: torch.Tensor, threshold: torch.Tensor, temperature: float) -> torch.Tensor:
    threshold = threshold.expand_as(x)
    y = torch.empty_like(x)
    left = x <= threshold
    right = ~left

    if left.any():
        x_left = x[left]
        threshold_left = threshold[left]
        num = x_left.div(threshold_left).pow(temperature)
        y[left] = num.div(num.add(1))

    if right.any():
        x_right = x[right]
        threshold_right = threshold[right]
        den = threshold_right.div(x_right).pow(temperature).add(1)
        y[right] = den.reciprocal()

    return y
