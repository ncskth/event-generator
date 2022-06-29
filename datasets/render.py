from typing import Callable
import torch


def render_shape(
    shape: Callable[[int, float, str], torch.Tensor],
    len: int,
    resolution: torch.Size,
    device: str,
    diameter: int = 80,
    shape_p: int = 1,
    bg_noise_p: float = 0.01,
    max_velocity: int = 1,
):
    """
    Draws a moving shape for `len` duration with maximum velocity of `max_velocity`.

    Arguments:
        shape (Callable): A function that generates a shape, expecting the shape size (int),
                          the bernouilli distribution probability (float), and the device
        len (int): The number of subsequent frames to generate
        resolution (torch.Size): The WxH resolution of the total frame. Should not be smaller than r
        device (str): The device on which the shape will be generated
        diameter (int): The diameter of the shape
        shape_p (float): The probability of drawing an event from a Bernouilli distribution
                         (1 = full shape, 0 = empty)
        bg_noise_p (float): The probability of drawing an event for added background noise from a
                         Bernouilli distribution (0 = no noise, 1 = full noise)
        max_velocity: The maximum allowed velocity between frames (1 = maximum moves one pixel)

    Returns:
        A tensor of size (len, resolution)

    """
    mask_r = diameter + 5
    images = torch.zeros(len, *resolution, dtype=torch.bool, device=device)
    labels = torch.zeros(len, 2)
    x = torch.randint(low=diameter, high=resolution[0] - diameter, size=(1,)).item()
    y = torch.randint(low=diameter, high=resolution[1] - diameter, size=(1,)).item()
    deltas = torch.distributions.Normal(loc=0, scale=0.2)
    velocity = torch.rand((2,), device=device)
    for i in range(images.shape[0]):
        x = x + velocity[0]
        y = y + velocity[1]
        if x < diameter or x > resolution[0] - diameter:
            velocity[0] = -1 * velocity[0]
        if y < diameter or y > resolution[1] - diameter:
            velocity[1] = -1 * velocity[1]
        velocity = (velocity + deltas.sample((2,)).to(device)).clip(
            -max_velocity, max_velocity
        )
        x = x.clip(mask_r, resolution[0] - mask_r - 1)
        y = y.clip(mask_r, resolution[1] - mask_r - 1)
        x_min = int(x.round() - diameter)
        x_max = x_min + diameter
        y_min = int(y.round() - diameter)
        y_max = y_min + diameter

        # Apply noise
        noise = (
            torch.distributions.Bernoulli(probs=bg_noise_p)
            .sample(images[i].shape)
            .to(device)
        )
        images[i] += noise.bool()

        # Fill in shape
        img = shape(diameter, p=shape_p, device=device)
        images[i, x_min:x_max, y_min:y_max] += img.bool()
        labels[i] = torch.tensor([(x_min + x_max) // 2, (y_min + y_max) // 2])

    return images.unsqueeze(1).float(), labels.unsqueeze(1)
