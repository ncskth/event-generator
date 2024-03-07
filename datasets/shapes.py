import torch
import torchvision


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


def triangle(size, p, device):
    res = torchvision.transforms.Resize((size, size), antialias=True)
    r = size
    one_sided = torch.tril(torch.ones(r, r, device=device))
    two_sided = torch.concat([one_sided[:-1], one_sided.flip(0)])
    space1 = torch.zeros((r * 2 - 1, 7 * r // 10), device=device)
    space2 = torch.zeros((r * 2 - 1, 7 * r // 10), device=device)
    tri = torch.concat([space1, two_sided, space2], dim=1)
    resized = res(tri.unsqueeze(0)).squeeze()
    return resized / resized.max()


def square(r, p, device):
    return torch.distributions.Bernoulli(probs=p).sample((r, r)).to(device)
