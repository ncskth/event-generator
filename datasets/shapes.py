import torch


def gaussian_mask(r, min, max, dist, device):
    width = 2 * r + 1
    g = (r - torch.arange(0, width, 1, device=device)) ** 2
    grid = g + g.unsqueeze(0).T
    img = torch.zeros(width, width, device=device)
    img = torch.where(
        (grid < max) & (grid > min), dist.sample((width, width)).to(device), img
    )
    return img.bool()


def circle(size, p, device):
    r = size / 2
    g = (r - 0.5 - torch.arange(0, size, 1, device=device)) ** 2
    grid = g + g.unsqueeze(0).T
    img = torch.zeros(size, size, device=device)
    dist = torch.distributions.Bernoulli(probs=p).sample((size, size)).to(device)

    return torch.where(grid < r**2, dist, img)


def triangle(r, p, device):
    r = int(round(r / 2) * 2)

    mid_1 = r // 2
    outer_1 = torch.distributions.Bernoulli(probs=p).sample((r, r)).to(device).bool()
    outer_full_1 = (
        torch.distributions.Bernoulli(probs=1).sample((r, r)).to(device).bool()
    )
    outer_left_1 = outer_1[:mid_1, :mid_1].tril(0).flip(1).repeat_interleave(2, 1)

    outer_full_1[:mid_1] &= outer_left_1
    outer_full_1[mid_1:] &= outer_left_1.flip(0)

    return outer_full_1.float()


def square(r, p, device):
    return torch.distributions.Bernoulli(probs=p).sample((r, r)).to(device)
