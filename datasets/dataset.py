from pathlib import Path

import torch


class ShapeDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str,
        t: int = 40,
        train: bool = True,
        pose_offset: torch.Tensor = torch.Tensor([0, 0]),
        pose_delay: int = 0,
        frames_per_file: int = 128,
        stack: int = 1,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.files = list(Path(root).glob("*.dat")) + list(Path(root).glob("**/*.dat"))
        split = int(0.8 * len(self.files))
        if train:
            self.files = self.files[:split]
        else:
            self.files = self.files[split:]
        self.stack = stack
        self.t = t + self.stack - 1
        self.pose_delay = int(pose_delay)
        self.chunks = frames_per_file // (2 * t + self.pose_delay)
        self.pose_offset = pose_offset.to(device)
        self.device = device
        assert len(self.files) > 0, f"No data files in given root '{root}'"

    def _stack_frames(self, frames):
        if self.stack > 1:
            offsets = [frames[: -self.stack + 1]] + [
                frames[x : self.t - self.stack + x + 1] for x in range(1, self.stack)
            ]
            return torch.stack(offsets, dim=1).squeeze()
        else:
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
