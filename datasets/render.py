from dataclasses import dataclass, field
from typing import Callable, Optional
import math
import numpy as np
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


def shear_tensor(image, shear_angle, shear):
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

    shear_image = transform(torch.unsqueeze(image, dim=0))
    shear_image = shear_image[0]

    non_zero_indices = torch.nonzero(shear_image)
    if non_zero_indices.shape[0] == 0:  # Allow for zero shear
        return image

    x_min = torch.min(non_zero_indices[:, 0])
    y_min = torch.min(non_zero_indices[:, 1])
    x_max = torch.max(non_zero_indices[:, 0])
    y_max = torch.max(non_zero_indices[:, 1])

    return shear_image[x_min:x_max, y_min:y_max]


ZERO_DISTRIBUTION = torch.distributions.Categorical(torch.tensor([1]))


def blit_shape(shape, bg, x, y, device):
    width = shape.shape[0]
    height = shape.shape[1]
    offset_x = torch.round(x) - x
    offset_y = torch.round(y) - y
    x_lin = torch.linspace(-1, 1, height).to(device) + offset_x / (width / 2)
    y_lin = torch.linspace(-1, 1, width).to(device) + offset_y / (height / 2)
    coo = (
        torch.stack(torch.meshgrid(x_lin, y_lin, indexing="xy"), -1)
        .unsqueeze(0)
        .float()
    )
    sampled_img = torch.nn.functional.grid_sample(
        shape.unsqueeze(0).unsqueeze(0).float(),
        coo,
        align_corners=False,
        padding_mode="zeros",
    ).squeeze()
    f = lambda x: round(x.item())  # round(round(x.item(), 1))# + 1e-5)
    left_x = f(x - width // 2 + offset_x)
    right_x = min(left_x + width, bg.shape[0])
    left_y = f(y - width // 2 + offset_y)
    right_y = min(left_y + height, bg.shape[1])
    bg[left_x:right_x, left_y:right_y] = sampled_img[
        : right_x - left_x, : right_y - left_y
    ]
    return offset_x, offset_y, left_x, left_y, sampled_img


@dataclass
class RenderParameters:
    """
    Parameters for rendering a shape

    resolution (torch.Size): The WxH resolution of the total frame
    length (int): The number of frames to generate
    device (str): The device on which the shape will be generated
    scale(Boolean): If true, the scale will vary
    translation(Boolean): If true, translation is applied
    rotate(Boolean): If true, rotation is applied
    shear(Boolean): If true, shear is applied

    shape_density (float): The probability of sampling the underlying shape contour. Default = 1
    event_density (float): The probability of sampling from the diff between two frames. Default = 1
    bg_noise_density (float): The probability of sampling from the background noise. Default = 0.002
    polarity (bool): If true, the shape will be rendered with polarity, output as PxXxY. Default = True
    """

    resolution: torch.Size
    length: int = 128
    event_density: float = 1
    shape_density: float = 1
    bg_noise_density: float = 0.001
    shape_density: float = 1
    polarity: bool = True
    warmup_steps: int = 5

    upsampling_factor: int = 8
    upsampling_cutoff: float = None
    device: str = "cuda"

    transformation_velocity_max: float = 1
    transformation_velocity_distribution: Callable[[float], torch.distributions.Distribution] = lambda x: torch.distributions.Normal(0, x)

    translate: bool = False
    translate_start_x: float = None
    translate_start_y: float = None
    translate_velocity_delta: Callable[[int], torch.Tensor] = None
    translate_velocity_max: float = field(init=False)
    translate_velocity_scale: float = 1
    translate_velocity_start: torch.Tensor = None
    scale: bool = False
    scale_start: float = None
    scale_velocity_delta: Callable[[int], torch.Tensor] = None
    scale_velocity_scale: float = 0.002
    scale_velocity_max: float = field(init=False)
    scale_velocity_start: float = None
    rotate: bool = False
    rotate_start: float = None
    rotate_velocity_delta: Callable[[int], torch.Tensor] = None
    rotate_velocity_scale: float = 0.2
    rotate_velocity_max: float = field(init=False)
    rotate_velocity_start: float = None
    shear: bool = False
    shear_start: float = None
    shear_max: float = 30
    shear_velocity_delta: Callable[[int], torch.Tensor] = None
    shear_velocity_scale: float = 0.2
    shear_velocity_max: float = field(init=False)
    shear_velocity_start: float = None

    def __post_init__(self):
        if self.upsampling_cutoff is None:
            self.upsampling_cutoff = 1 / self.upsampling_factor
        for attr in ["translate", "scale", "rotate", "shear"]:
            max_velocity = getattr(self, f"transformation_velocity_max") * getattr(self, f"{attr}_velocity_scale")
            setattr(self, f"{attr}_velocity_max", max_velocity)
            if getattr(self, attr + "_velocity_delta") is None:
                if getattr(self, attr):
                    setattr(
                        self,
                        f"{attr}_velocity_delta",
                        lambda s: self.transformation_velocity_distribution(getattr(self, f"{attr}_velocity_scale")).sample((s,))
                    )
                else:
                    setattr(
                        self,
                        f"{attr}_velocity_delta",
                        lambda s: ZERO_DISTRIBUTION.sample((s,)),
                    )


class IAFSubtractReset(torch.nn.Module):

    def __init__(self, cutoff: float):
        super().__init__()
        self.cutoff = cutoff

    def forward(self, x, state=None):
        if state is None:
            state = torch.zeros_like(x)
        v_new = state + x
        # compute new positive and negative spikes
        z_pos = v_new > self.cutoff
        z_neg = v_new < -self.cutoff
        # compute reset
        v_new = v_new - z_pos * self.cutoff + z_neg * self.cutoff
        return torch.stack([z_pos, z_neg]), v_new


def render_shape(
    shape_fn: Callable[[int, float, str], torch.Tensor],
    p: RenderParameters,
):
    """
    Draws a moving shape for `length` duration with maximum velocity of `max_velocity`.
    Arguments:
        shape_fn (Callable[[int, float, str], torch.Tensor]): The function generating the shape
        p (RenderParameters): The parameters for rendering the shape
    Returns:
        A tensor of size (lengh, resolution)
    """
    assert (
        p.resolution[0] >= 256 and p.resolution[1] >= 256
    ), "Resolution must be >= 256"

    bg_noise_dist = torch.distributions.Bernoulli(probs=p.bg_noise_density)
    event_dist = torch.distributions.Bernoulli(probs=p.event_density)

    mask_r = 5
    images = torch.zeros(p.length, 2, *p.resolution, dtype=torch.bool, device=p.device)
    labels = torch.zeros(p.length, 2)
    neuron_population = IAFSubtractReset(p.upsampling_cutoff)
    neuron_state = None
    current_image = None
    previous_image = None

    angle = p.rotate_start if p.rotate_start is not None else 0
    angle_velocity = (
        p.rotate_velocity_start * p.rotate_velocity_scale
        if p.rotate_velocity_start is not None
        else 0
    )
    shear = p.shear_start if p.shear_start is not None else 0
    shear_velocity = (
        p.shear_velocity_start * p.shear_velocity_scale
        if p.shear_velocity_start is not None
        else 0
    )
    scale = p.scale_start if p.scale_start is not None else 0
    scale_velocity = (
        p.scale_velocity_start * p.scale_velocity_scale
        if p.scale_velocity_start is not None
        else 0
    )

    min_resolution = torch.as_tensor(min(p.resolution[0], p.resolution[1]))
    min_size = (0.05 * min_resolution).int().to(p.device)
    max_size = (0.6 * min_resolution).int().to(p.device)
    resolution_upscaled = torch.as_tensor(p.resolution) * p.upsampling_factor

    # Initialize starting x, y
    x = (
        p.translate_start_x * p.upsampling_factor
        if p.translate_start_x is not None
        else torch.randint(
            low=int(scale) // 2 + mask_r * p.upsampling_factor,
            high=resolution_upscaled[0]
            - int(scale) // 2
            - mask_r * p.upsampling_factor,
            size=(1,),
            device=p.device,
        )
    )
    y = (
        p.translate_start_y * p.upsampling_factor
        if p.translate_start_y is not None
        else torch.randint(
            low=int(scale) // 2 + mask_r * p.upsampling_factor,
            high=resolution_upscaled[1]
            - int(scale) // 2
            - mask_r * p.upsampling_factor,
            size=(1,),
            device=p.device,
        )
    )
    if p.translate:
        trans_velocity = (
            (
                (torch.rand((2,), device=p.device) - 1.5) * p.translate_velocity_max
            )
            if p.translate_velocity_start is None
            else p.translate_velocity_start * p.translate_velocity_scale
        )
    else:
        trans_velocity = torch.zeros((2,), device=p.device)
    # else:
    #     trans_velocity = torch.zeros((2,), device=p.device)
    #     x = (
    #         resolution_upscaled[0] // 2
    #         if p.translate_start_x is None
    #         else p.translate_start_x
    #     )
    #     y = (
    #         resolution_upscaled[1] // 2
    #         if p.translate_start_y is None
    #         else p.translate_start_y
    #     )

    # Initialize scale
    if p.scale:
        scale = (
            (torch.rand((1,), device=p.device) * (max_size - min_size) + min_size)
            if p.scale_start is None
            else p.scale_start
        )
    else:
        scale = min_size + (max_size - min_size) // 2

    # Initialize rotation
    if p.rotate:
        angle = (
            torch.randint(low=0, high=360, size=(1,)).item()
            if p.rotate_start is None
            else p.rotate_start
        )

    # Initialize shear
    if p.shear:
        shear = (
            torch.randint(low=0, high=p.shear_max, size=(1,)).item()
            if p.shear_start is None
            else p.shear_start
        )

    # Loop timesteps
    for i in range(-p.warmup_steps - 1, images.shape[0]):
        # Fill in shape
        img = shape_fn(
            int(scale * p.upsampling_factor), p=p.shape_density, device=p.device
        )

        # Translate
        x = x + trans_velocity[0] 
        y = y + trans_velocity[1] 
        x = x.clip(
            int(img.size()[0] * np.sqrt(2) / 2) + mask_r * p.upsampling_factor,
            resolution_upscaled[0]
            - int(img.size()[0] * np.sqrt(2) / 2)
            - mask_r * p.upsampling_factor
            - 1,
        )
        y = y.clip(
            int(img.size()[1] * np.sqrt(2) / 2) + mask_r * p.upsampling_factor,
            resolution_upscaled[1]
            - int(img.size()[1] * np.sqrt(2) / 2)
            - mask_r * p.upsampling_factor
            - 1,
        )
        trans_velocity = (
            trans_velocity + p.translate_velocity_delta(2).to(p.device)
        ).clip(-p.translate_velocity_max, p.translate_velocity_max)
        # Rotate
        angle = angle + angle_velocity
        angle_velocity = (
            angle_velocity
            + p.rotate_velocity_delta(1).to(p.device) * 180 / torch.pi
        ).clip(-p.rotate_velocity_max, p.rotate_velocity_max)
        img = rotate_tensor(img, float(angle))
        # Shear
        if shear >= p.shear_max or shear == -p.shear_max:
            shear_velocity = -1 * shear_velocity
        shear_velocity = (shear_velocity + p.shear_velocity_delta(1).to(p.device)).clip(
            -p.shear_velocity_max, p.shear_velocity_max
        )
        shear = (shear + shear_velocity[0]).clip(-p.shear_max, p.shear_max)
        img = shear_tensor(img, 0, shear)

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
            x <= mask_r * p.upsampling_factor
            or x >= resolution_upscaled[0] - mask_r * p.upsampling_factor
        ):
            trans_velocity[0] *= -1
            x.clip(
                mask_r * p.upsampling_factor,
                resolution_upscaled[0] - mask_r * p.upsampling_factor,
            )
        # Flip vertical translational velocity if shape is at boundary
        if (
            y <= mask_r * p.upsampling_factor
            or y >= resolution_upscaled[1] - mask_r * p.upsampling_factor
        ):
            trans_velocity[1] *= -1
            y.clip(
                mask_r * p.upsampling_factor,
                resolution_upscaled[1] - mask_r * p.upsampling_factor,
            )

        # Scale
        if p.scale:
            scale_velocity = scale_velocity + p.scale_velocity_delta(1).to(
                p.device
            ).clip(-p.scale_velocity_max, p.scale_velocity_max)
            scale = scale + scale_velocity * p.upsampling_factor

        # Blit image onto frame
        x_center, y_center = torch.tensor(img.shape) / 2
        x_min = x - x_center
        x_min_cropped = int(max(0, x_min.round()))
        y_min = y - y_center
        y_min_cropped = max(0, int(y_min))
        x_max = int(min(round(x_min_cropped + img.shape[0]), resolution_upscaled[0]))
        y_max = int(min(round(y_min_cropped + img.shape[1]), resolution_upscaled[0]))
        current_image = torch.zeros(*resolution_upscaled, device=p.device)
        blit_shape(img, current_image, x, y, p.device)

        # Downsample and compare
        # Note, we ignore the first image because we're taking the difference
        if previous_image is not None:
            downsample = lambda x: torch.nn.functional.interpolate(
                x.unsqueeze(0).unsqueeze(0),
                scale_factor=1 / p.upsampling_factor,
                mode="bilinear",
                antialias=False,
            )
            # Take the diff
            prev_down = downsample(previous_image)
            curr_down = downsample(current_image)
            downsampled_diff = (prev_down - curr_down).squeeze()
            noise_mask = event_dist.sample((downsampled_diff.shape)).bool().to(p.device)
            ch1, neuron_state = neuron_population(downsampled_diff, neuron_state)

            # Assign channels after warmup
            if i > 0:
                images[i] = ch1.bool() & noise_mask
                labels[i] = (
                    torch.tensor(
                        [(x_min_cropped + x_max) // 2, (y_min_cropped + y_max) // 2]
                    )
                    / p.upsampling_factor
                )

        previous_image = current_image

        if p.device.startswith("cuda"):
            torch.cuda.empty_cache()

    # Add noise
    images += bg_noise_dist.sample(images.shape).to(p.device).bool()
    return images.float().clip(0, 1), labels.unsqueeze(1)


if __name__ == "__main__":
    import shapes

    # p = RenderParameters(resolution=torch.Size([256, 256]))
    p = RenderParameters(
        torch.Size((300, 300)),
        20,
        scale=True,
        # translate=,
        # scale_start=50,
        rotate_start=10,
        transformation_velocity_max=0.2,
        scale_velocity_start=1,
        # scale_velocity_start=1.6,
        upsampling_factor=4,
        upsampling_cutoff=1 / 4,
        bg_noise_density=0,
        scale_velocity_delta=lambda s: torch.tensor([0]),
    )
    render_shape(shapes.triangle, p)
