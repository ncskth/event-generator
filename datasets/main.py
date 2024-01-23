import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import traceback
from typing import List, NamedTuple, Optional

import torch
import tqdm
from .render import render_shape
from .shapes import *


class DatasetParameters(NamedTuple):
    resolution: torch.Size
    event_density: float
    bg_density: float
    shape_density: float = 1
    event_density: float = 1

    upsampling_factor: int = 8
    upsampling_cutoff: int = 16
    bg_files: Optional[List[str]] = None
    device: str = "cuda"
    length: int = 128

    transformation: bool = False
    scale: bool = False
    rotate: bool = False
    shear: bool = False


def superimpose_data(file, images, resolution, device):
    _, _, frames, _, _, _, _ = torch.load(file, map_location=device)
    # Reduce polarity
    frames = frames.sum(-1, keepdim=True)
    # Crop
    frames = frames[:, : resolution[0], : resolution[1]]
    # Permute to TCHW
    frames = frames.permute(0, 3, 2, 1)
    # Normalize
    images = images.clip(0, 1)
    frames = frames.clip(0, 1)
    return (images + frames).clip(0, 1)


def render_shapes(p: DatasetParameters):
    shapes = []
    labels = []
    for fn in [circle, square, triangle]:
        s, l = render_shape(
            fn,
            len=p.length,
            resolution=p.resolution,
            shape_p=p.shape_density,
            bg_noise_p=p.bg_density,
            event_p=p.event_density,
            device=p.device,
            scale_change=p.scale,
            trans_change=p.transformation,
            rotate_change=p.rotate,
            skew_change=p.shear,
            upscale_factor=p.upsampling_factor,
            upscale_cutoff=p.upsampling_cutoff,
        )
        shapes.append(s)
        labels.append(l)

    images = torch.stack(shapes).sum(0)
    labels = torch.stack(labels).permute(1, 0, 2, 3)
    return images, labels


def render_points(output_folder, index, p: DatasetParameters):
    filename = output_folder / f"{index}.dat"
    try:
        if p.bg_files is not None:
            images, labels = render_shapes(p)
            images = superimpose_data(
                p.bg_files[index % len(p.bg_files)], images, p.resolution, p.device
            )

        t = [images.clip(0, 1).to_sparse(), labels]
        torch.save(t, filename)
    except Exception as e:
        print(e)
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Render dataset")
    parser.add_argument("root", type=str, help="Path to output directory")
    parser.add_argument(
        "root_bg",
        type=str,
        default=None,
        help="Location of dataset to use as background",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed to initialize random dataset mapping",
    )
    parser.add_argument(
        "--translation",
        type=bool,
        action="store_true",
    )
    parser.add_argument(
        "--scaling",
        type=bool,
        action="store_true",
    )
    parser.add_argument(
        "--rotation",
        type=bool,
        action="store_true",
    )
    parser.add_argument(
        "--shearing",
        type=bool,
        action="store_true",
    )
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    n = torch.arange(2000)
    threads = 12
    event_ps = [0.01, 0.1, 0.2, 0.4, 0.8, 1.0]
    resolution = (300, 300)
    device = "cuda"
    root_folder = Path(args.root)
    if not root_folder.exists():
        root_folder.mkdir()

    bg_folder = Path(args.root_bg)
    if bg_folder.exists():
        bg_files = list(bg_folder.glob("*.dat"))
        sorted(bg_files)
    else:
        bg_files = None

    # Permutations of transformations
    transformation_combinations = [torch.tensor([args.translation, args.scaling, args.rotation, args.shearing])]

    with tqdm.tqdm(
        total=len(event_ps) * n.numel() * len(transformation_combinations)
    ) as bar:
        with ThreadPoolExecutor(max_workers=threads) as ex:
            futures = []
            for p in event_ps:
                for comb in transformation_combinations:
                    parameters = DatasetParameters(
                        resolution=resolution,
                        shape_density=1,
                        bg_density=0.001,
                        event_density=p,
                        transformation=comb[0],
                        scale=comb[1],
                        rotate=comb[2],
                        shear=comb[3],
                    )
                    combination_name = (
                        str(comb.tolist()).replace(", ", "").replace("True", "1")[1:-1]
                    )
                    output_folder = root_folder / f"{p:.1}-{combination_name}"
                    for i in n:
                        if not output_folder.exists():
                            output_folder.mkdir()
                        f = ex.submit(render_points, output_folder, i, parameters)
                        futures.append(f)
            for f in as_completed(futures):
                bar.update(1)
