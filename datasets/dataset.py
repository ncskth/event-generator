from pathlib import Path
from random import shuffle
from typing import Optional
import re

import torch


class ShapeDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str,
        t: int = 40,
        train: bool = True,
        file_filter: Optional[str] = None,
        pose_offset: torch.Tensor = torch.Tensor([0, 0]),
        pose_delay: int = 0,
        frames_per_file: int = 128,
        stack: int = 1,
        sum_frames: bool = False,
        device: str = "cpu",
        shuffle_files: bool = False
    ) -> None:
        super().__init__()
        self.files = []
        subdirs = Path(root).glob("*")
        for d in subdirs:
            self.files.extend(ShapeDataset._get_subdir_files(d, train, shuffle_files))
        if file_filter:
            regex = re.compile(file_filter)
            self.files = [f for f in self.files if regex.search(str(f)) is not None]
        self.stack = stack
        self.sum_frames = sum_frames
        self.t = t + self.stack - 1
        self.pose_delay = int(pose_delay)
        self.chunks = frames_per_file // (2 * t + self.pose_delay)
        self.pose_offset = pose_offset.to(device)
        self.device = device
        assert len(self.files) > 0, f"No data files in given root '{root}'"

    @staticmethod
    def _get_subdir_files(d, train, shuffle_files):
        files = list(d.glob("*.dat"))
        if shuffle_files:
            shuffle(files)
        else:
            files = sorted(files)
        split = int(0.8 * len(files))
        if train:
            return files[:split]
        else:
            return files[split:]

    def _stack_frames(self, frames):
        if self.stack > 1:
            offsets = [frames[: -self.stack + 1]] + [
                frames[x : self.t - self.stack + x + 1] for x in range(1, self.stack)
            ]
            frames = torch.stack(offsets, dim=1)
            if self.sum_frames:
                frames = frames.sum(1)
            else:
                frames = frames.flatten(1, 2)
        return frames

    def __getitem__(self, index):
        filename = self.files[index // self.chunks]
        frames, poses = torch.load(filename, map_location=self.device)
        frames = frames.to_dense()
        chunk = index % self.chunks
        start = chunk * self.t
        mid = start + self.t
        end = mid + self.t
        warmup_tensor = frames[start:mid].float().clip(0, 1)
        actual_tensor = frames[mid:end].float().clip(0, 1)
        delayed_poses = poses[mid + self.pose_delay : end + self.pose_delay]
        offset_poses = delayed_poses + self.pose_offset
        return (
            self._stack_frames(warmup_tensor),
            self._stack_frames(actual_tensor),
            offset_poses.squeeze()[self.stack - 1 :],
        )

    def __len__(self):
        return len(self.files) * self.chunks


if __name__ == "__main__":
    import tqdm

    d = ShapeDataset("/mnt/raid0a/shapes/", pose_delay=3, stack=9, train=True)
    print(len(d))
    bar = tqdm.tqdm(range(len(d)))
    for i in bar:
        bar.set_description(f"{i}, {len(d[i])}")
