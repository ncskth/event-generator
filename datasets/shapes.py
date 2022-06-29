from typing import Callable
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
    width = 1.75 * size
    r = size // 2
    g = (r - torch.arange(0, size, 1, device=device)) ** 2
    grid = g + g.unsqueeze(0).T
    img = torch.zeros(size, size, device=device)
    dist = torch.distributions.Bernoulli(probs=p).sample((size, size)).to(device)
    outer_ring = r**2 - width
    inner_ring = r**2 - width * 3
    img = torch.where(
        (grid < outer_ring + width) & (grid > outer_ring - width), dist, img
    )
    return img


def square(r, p, device, width=3):
    size = r
    outer = torch.distributions.Bernoulli(probs=p).sample((size, size)).to(device)
    inner = torch.zeros(size - width * 2, size - width * 2, device=device)
    outer[width : r - width, width : r - width] = inner
    return outer


def triangle(r, p, device, thickness=5):
    mid = r // 2
    outer = torch.distributions.Bernoulli(probs=p).sample((r, r)).to(device).bool()

    # Outer
    outer_left = outer[:mid, :mid].tril(0).flip(1).repeat_interleave(2, 1)
    outer[:mid] &= outer_left
    outer[mid:] &= outer_left.flip(0)

    # Inner
    inner_left = (
        torch.ones(mid - thickness, mid - thickness, device=device)
        .tril(2)
        .bool()
        .flip(0)
        .repeat_interleave(2, 1)
    )
    outer[thickness:mid, thickness : r - thickness] &= inner_left
    outer[mid:-thickness, thickness : r - thickness] &= inner_left.flip(0)

    return outer
