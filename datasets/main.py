import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List

import torch
import tqdm
from render import render_shape_improved
from shapes import *


def superimpose_data(files, images, resolution, device):
    file = files[torch.randint(0, len(files), (1,)).item()]
    _, _, frames, _, _, _, _ = torch.load(file, map_location=device)
    # Reduce polarity
    frames = frames.sum(-1, keepdim=True)
    # Crop
    frames = frames[:, : resolution[0], : resolution[1]]
    # Permute to TCHW
    frames = frames.permute(0, 3, 2, 1)
    return images + frames


def render_points(
    output_folder,
    index,
    resolution,
    p: float,
    bg_p: float,
    device: str,
    bg_files: List[str] = None,
):
    filename = output_folder / f"{index}.dat"
    try:
        shapes = []
        labels = []
        for fn in [circle_improved, square_improved, triangle_improved]:
            s, l = render_shape_improved(
                fn,
                len=128,
                resolution=resolution,
                shape_p=p,
                bg_noise_p=bg_p,
                device=device,
                scale_change=True,
                trans_change=True,
                rotate_change=False,
            )
            shapes.append(s)
            labels.append(l)

        images = torch.stack(shapes).sum(0)
        labels = torch.stack(labels).permute(1, 0, 2, 3)
        if bg_files is not None:
            images = superimpose_data(bg_files, images, resolution, device)

        t = [images.to_sparse(), labels]
        torch.save(t, filename)
    except Exception as e:
        print(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Render dataset")
    parser.add_argument("root", type=str, help="Path to output directory")
    parser.add_argument(
        "root_bg",
        type=str,
        default=None,
        help="Location of dataset to use as background",
    )
    args = parser.parse_args()

    n = torch.arange(2000)
    threads = 12
    #ps = torch.linspace(0, 0.9, 10)
    ps = [0.8]
    resolution = (256, 256)
    device = "cuda"
    root_folder = Path(args.root)
    if not root_folder.exists():
        root_folder.mkdir()

    bg_folder = Path(args.root_bg)
    if bg_folder.exists():
        bg_files = list(bg_folder.glob("*.dat"))
    else:
        bg_files = None

    with tqdm.tqdm(total=len(ps) * n.numel()) as bar:
        with ThreadPoolExecutor() as ex:
            futures = []
            for p in ps:
                for i in n:
                    output_folder = root_folder / f"{p:.1}"
                    if not output_folder.exists():
                        output_folder.mkdir()
                    f = ex.submit(
                        render_points,
                        output_folder,
                        i,
                        resolution=resolution,
                        p=p,
                        bg_p=0.002,
                        device=device,
                        bg_files=bg_files,
                    )
                    futures.append(f)
            for f in as_completed(futures):
                bar.update(1)
