from typing import Callable, Optional
import math
import numpy as np
import norse
import torch
import torchvision


def events_to_frames(frames, polarity: bool = False):
    if len(frames.shape) == 3:
        frames = frames.unsqueeze(-1).repeat(1, 1, 1, 3)
    else:
        if not polarity:
            frames = frames.abs().sum(-1)
        elif polarity:
            frames = torch.concat(
                [frames, torch.zeros(*frames.shape[:-1], 1, device=frames.device)],
                dim=-1,
            )
    frames = ((frames / frames.max()) * 255).int()
    return frames


def rotate_tensor(input, x):
    rotated_input = torchvision.transforms.functional.rotate(
        torch.unsqueeze(input, dim=0),
        x,
        expand=True,
        fill=0,
        interpolation=torchvision.transforms.InterpolationMode.NEAREST,
    )
    return rotated_input[0]


def skew_tensor(image, shear_angle, shear):
    pad = int(image.size()[0] * 2)

    y_shear = shear * math.sin(math.pi * shear * shear_angle / 180)
    x_shear = shear * math.cos(math.pi * shear * shear_angle / 180)

    # define the transformation to be applied to images
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Pad(pad),
            torchvision.transforms.RandomAffine(
                degrees=0,
                shear=[x_shear, x_shear + 0.01, y_shear, y_shear + 0.01],
                fill=0,
                interpolation=torchvision.transforms.InterpolationMode.NEAREST,
            ),
        ]
    )

    skew_image = transform(torch.unsqueeze(image, dim=0))
    skew_image = skew_image[0]

    non_zero_indices = torch.nonzero(skew_image)
    if non_zero_indices.shape[0] == 0:  # Allow for zero shear
        return image

    x_min = torch.min(non_zero_indices[:, 0])
    y_min = torch.min(non_zero_indices[:, 1])
    x_max = torch.max(non_zero_indices[:, 0])
    y_max = torch.max(non_zero_indices[:, 1])

    return skew_image[x_min:x_max, y_min:y_max]


def render_shape(
    shape_fn: Callable[[int, float, str], torch.Tensor],
    len: int,
    resolution: torch.Size,
    device: str,
    scale_change: bool,
    trans_change: bool,
    rotate_change: bool,
    skew_change: bool,
    shape_p: float = 1,
    event_p: float = 0.8,
    bg_noise_p: float = 0.001,
    max_trans_change: float = 1,
    max_scale_change: float = 1,
    max_shear_change: float = 1,
    max_angle_change: float = 1,
    max_shear_value: float = 40,
    upscale_factor: int = 8,
    upscale_cutoff: Optional[int] = None,
):
    """
    Draws a moving shape for `len` duration with maximum velocity of `max_velocity`.
    Arguments:
        shape_fn (Callable[[int, float, str], torch.Tensor]): The function generating the shape
        len (int): The number of subsequent frames to generate
        resolution (torch.Size): The WxH resolution of the total frame. Should not be smaller than r
        device (str): The device on which the shape will be generated
        scale_change (Boolean): If true, the scale will vary
        trans_change (Boolean): If true, translation is applied
        rotate_change (Boolean): If true, rotation is applied
        skew_change (Boolean): If true, skew is applied
        shape_p (float): The density of the underlying shape, drawn from a Bernouilli distribution
                         (1 = full shape, 0 = empty)
        event_p (float): The density of the events, calculated from the difference of two subsequent underlying shapes,
                         drawn from a Bernouilli distribution (1 = full, 0 = empty)
        bg_noise_p (float): The probability of drawing an event for added background noise from a
                         Bernouilli distribution (0 = no noise, 1 = full noise)
        max_trans_change: The maximum allowed translational velocity between frames (1 = maximum moves one pixel)
        max_scale_change: The max change in the scale of the shape between frames
        max_shear_change: The max change in shearing between frames
        max_angle_change: The maximum allowed velocity between frames (1 = maximum moves one pixel)
        max_shear_value (float): The maximum allowed shearing in pixels relative to the upscaling
        upscale_factor (int): Factor from which the underlying events are sampled for increased precision
        upscale_cutoff (int): The number of events of the upsampled picture that triggers an event in the downsampled image. Defaults to 1/upscale_factor
    Returns:
        A tensor of size (len, resolution)
    """
    assert resolution[0] >= 256 and resolution[1] >= 256, "Resolution must be >= 256"

    bg_noise_dist = torch.distributions.Bernoulli(probs=bg_noise_p)
    event_noise_dist = torch.distributions.Bernoulli(probs=event_p)
    if upscale_cutoff is None:
        upscale_cutoff = 1 / upscale_factor
    # max_shear_value = max_shear_value * upscale_factor

    mask_r = 5
    images = torch.zeros(len, 2, *resolution, dtype=torch.bool, device=device)
    neuron_on = norse.torch.LIFBoxCell(
        p=norse.torch.LIFBoxParameters(tau_mem_inv=100, v_th=upscale_cutoff)
    )
    neuron_off = norse.torch.LIFBoxCell(
        p=norse.torch.LIFBoxParameters(tau_mem_inv=100, v_th=upscale_cutoff)
    )
    neuron_state = [None, None]
    labels = torch.zeros(len, 2)
    current_image = None
    previous_image = None

    angle = 0
    angle_velocity = 0
    shear = 0
    shear_velocity = 0
    scale = 0
    scale_velocity = 0

    min_resolution = torch.as_tensor(min(resolution[0], resolution[1]))
    min_size = (0.05 * min_resolution).int().to(device)
    max_size = (0.6 * min_resolution).int().to(device)
    resolution_upscaled = torch.as_tensor(resolution) * upscale_factor

    if scale_change:
        delta_scale = torch.distributions.Normal(loc=0, scale=0.1)
        scale = torch.rand((1,), device=device) * (max_size - min_size) + min_size
        scale_velocity = (torch.rand((1,), device=device) - 1) * max_scale_change * 0.5
    else:
        scale = min_size + (max_size - min_size) // 2

    if trans_change:
        deltas = torch.distributions.Normal(loc=0, scale=0.1)
        trans_velocity = (torch.rand((2,), device=device) - 0.5) * max_trans_change * 2
    else:
        trans_velocity = torch.zeros((2,), device=device)

    if rotate_change:
        angle = torch.randint(low=0, high=360, size=(1,)).item()
        angle_delta = torch.distributions.Normal(loc=0, scale=0.2)
        angle_velocity = (
            ((torch.rand((1,), device=device) - 1) * max_angle_change)
            .clip(-max_angle_change, max_angle_change)
            .to(device)
        )

    if skew_change:
        shear = torch.randint(low=0, high=20, size=(1,)).item()
        delta_shear = torch.distributions.Normal(loc=0, scale=0.6)
        shear_velocity = (torch.rand((1,), device=device) - 0.5) * max_shear_change * 2

    # Initialize starting x, y
    x = torch.randint(
        low=int(scale) // 2 + mask_r * upscale_factor,
        high=resolution_upscaled[0] - int(scale) // 2 - mask_r * upscale_factor,
        size=(1,),
        device=device,
    )
    y = torch.randint(
        low=int(scale) // 2 + mask_r * upscale_factor,
        high=resolution_upscaled[1] - int(scale) // 2 - mask_r * upscale_factor,
        size=(1,),
        device=device,
    )

    # Loop timesteps
    for i in range(-1, images.shape[0]):
        # Fill in shape
        img = shape_fn(int(scale * upscale_factor), p=shape_p, device=device)

        # Translate
        if trans_change:
            x = x + trans_velocity[0] * upscale_factor
            y = y + trans_velocity[1] * upscale_factor
            trans_velocity = (trans_velocity + deltas.sample((2,)).to(device)).clip(
                -max_trans_change, max_trans_change
            )
        # Rotate
        if rotate_change:
            rotated_img = rotate_tensor(img, angle)
            angle = angle + angle_velocity.item()
            angle_velocity = (
                angle_velocity + angle_delta.sample((1,)).to(device)
            ).clip(-max_angle_change, max_angle_change)
        else:
            rotated_img = img
        # Skew
        if skew_change:
            skewed_img = skew_tensor(rotated_img, 0, shear)
        else:
            skewed_img = rotated_img
        rotated_img = skewed_img
        if skew_change:
            # to keep the shear movement
            if shear == 20 or shear == -20:
                shear_velocity = -1 * shear_velocity

            shear_velocity = (
                shear_velocity + delta_shear.sample((1,)).to(device)
            ).clip(-max_shear_change, max_shear_change)
            shear = (shear + shear_velocity[0]).clip(-max_shear_value, max_shear_value)
        else:
            shear = 0

        # Flip scaling if size too big
        if scale >= max_size:
            scale = max_size
            scale_velocity = -1 * scale_velocity
        # Flip scaling if size too small
        if scale <= min_size:
            scale = min_size
            scale_velocity = -1 * scale_velocity
        # Flip horizontal translation velocity if shape is at boundary
        if (
            x <= mask_r * upscale_factor
            or x
            >= resolution_upscaled[0]
            - (rotated_img.shape[0] // 2)
            - mask_r * upscale_factor
        ):
            trans_velocity[0] *= -1
        # Flip vertical translational velocity if shape is at boundary
        if (
            y <= mask_r * upscale_factor
            or y
            >= resolution_upscaled[1]
            - (rotated_img.shape[1] // 2)
            - mask_r * upscale_factor
        ):
            trans_velocity[1] *= -1

        # Scale
        if scale_change:
            scale_velocity = scale_velocity + delta_scale.sample((1,)).to(device).clip(
                -max_scale_change
            ).clip(-max_scale_change, max_scale_change)
            scale = scale + scale_velocity

        # Update coordinates
        x = x.clip(
            int(rotated_img.size()[0] * np.sqrt(2) / 2) + mask_r * upscale_factor,
            resolution_upscaled[0]
            - int(rotated_img.size()[0] * np.sqrt(2) / 2)
            - mask_r * upscale_factor
            - 1,
        )
        y = y.clip(
            int(rotated_img.size()[1] * np.sqrt(2) / 2) + mask_r * upscale_factor,
            resolution_upscaled[1]
            - int(rotated_img.size()[1] * np.sqrt(2) / 2)
            - mask_r * upscale_factor
            - 1,
        )

        x_center, y_center = torch.tensor(rotated_img.shape) / 2
        x_min = x - x_center
        x_offset = (x_min.round() - x_min) / round(rotated_img.shape[0] / 2)
        x_min_cropped = max(0, int(x_min.round()))
        x_max = min(int(x_min_cropped + rotated_img.shape[0]), resolution_upscaled[0])
        y_min = y - y_center
        y_offset = (y_min.round() - y_min) / round(rotated_img.shape[1] / 2)
        y_min_cropped = max(0, int(y_min.round()))
        y_max = min(int(y_min_cropped + rotated_img.shape[1]), resolution_upscaled[0])

        x_lin = torch.linspace(-1, 1, rotated_img.shape[0]).to(device) + x_offset
        y_lin = torch.linspace(-1, 1, rotated_img.shape[1]).to(device) + y_offset
        coo = torch.stack(torch.meshgrid(x_lin, y_lin), -1).unsqueeze(0).float()
        sampled_img = torch.nn.functional.grid_sample(
            rotated_img.unsqueeze(0).unsqueeze(0), coo, align_corners=False
        ).squeeze()
        current_image = torch.zeros(*resolution_upscaled, device=device)
        current_image[x_min_cropped:x_max, y_min_cropped:y_max] = sampled_img

        # Note, we ignore the first image because we're taking the difference
        if previous_image is not None:
            # Take the diff
            diff = previous_image - current_image
            downsampled_diff = torch.nn.functional.interpolate(
                diff.unsqueeze(0).unsqueeze(0),
                scale_factor=1 / upscale_factor,
                mode="bilinear",
                antialias=False,
            )
            noise_mask = (
                event_noise_dist.sample((downsampled_diff.shape)).bool().to(device)
            )
            ch1, neuron_state[0] = neuron_on(downsampled_diff * 10, neuron_state[0])
            ch2, neuron_state[1] = neuron_off(downsampled_diff * -10, neuron_state[1])
            # Assign channels
            images[i, 0] = ch1.bool() & noise_mask
            images[i, 1] = ch2.bool() & noise_mask

            labels[i] = torch.tensor(
                [(x_min_cropped + x_max) // 2, (y_min_cropped + y_max) // 2]
            )

        previous_image = current_image

    # Add noise
    images += bg_noise_dist.sample(images.shape).to(device).bool()
    return images.float().clip(0, 1), labels.unsqueeze(1)


if __name__ == "__main__":
    from shapes import *

    for fn in [circle, square, triangle]:
        s, l = render_shape(
            fn,
            len=128,
            resolution=(300, 300),
            shape_p=0.8,
            bg_noise_p=0.002,
            device="cuda",
            scale_change=True,
            trans_change=True,
            rotate_change=True,
            skew_change=True,
        )
