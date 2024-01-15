import os
from os import path, mkdir
from typing import Dict, Union
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from FE import transforms_video
from FE.videoIter import VideoIter
from FE.load_model import load_feature_extractor


def build_transforms() :
    mean = [124 / 255, 117 / 255, 104 / 255]
    std = [1 / (0.0167 * 255)] * 3
    resize = 128, 171
    crop = 112
    res = transforms.Compose(
        [
            transforms_video.ToTensorVideo(),
            transforms_video.ResizeVideo(resize),
            transforms_video.CenterCropVideo(crop),
            transforms_video.NormalizeVideo(mean=mean, std=std),
        ])
    return res



def to_segments(data, n_segments= 32) :
    data = np.array(data)
    Segments_Features = []
    thirty2_shots = np.round(np.linspace(0, len(data) - 1, num=n_segments + 1)).astype(int)
    for ss, ee in zip(thirty2_shots[:-1], thirty2_shots[1:]):
        if ss == ee:
            temp_vect = data[min(ss, data.shape[0] - 1), :]
        else:
            temp_vect = data[ss:ee, :].mean(axis=0)

        temp_vect = temp_vect / np.linalg.norm(temp_vect)

        if np.linalg.norm(temp_vect) != 0:
            Segments_Features.append(temp_vect.tolist())

    return Segments_Features


class FeaturesWriter:
    def __init__(self, num_videos: int, chunk_size: int = 16):
        self.path = None
        self.dir = None
        self.data = None
        self.chunk_size = chunk_size
        self.num_videos = num_videos
        self.dump_count = 0

    def _init_video(self, video_name: str, dir: str):
        self.path = path.join(dir, f"{video_name}.txt")
        self.dir = dir
        self.data = {}

    def has_video(self):
        return self.data is not None

    def dump(self):
        self.dump_count += 1
        if not path.exists(self.dir):
            os.mkdir(self.dir)

        features = to_segments([self.data[key] for key in sorted(self.data)])
        with open(self.path, "w") as fp:
            for d in features:
                d = [str(x) for x in d]
                fp.write(" ".join(d) + "\n")

    def _is_new_video(self, video_name: str, dir: str):
        new_path = path.join(dir, f"{video_name}.txt")
        if self.path != new_path and self.path is not None:
            return True

        return False

    def store(self, feature: Union[Tensor, np.ndarray], idx: int):
        self.data[idx] = list(feature)

    def write(self, feature: Union[Tensor, np.ndarray], video_name: str, idx: int, dir: str):
        if not self.has_video():
            self._init_video(video_name, dir)

        if self._is_new_video(video_name, dir):
            self.dump()
            self._init_video(video_name, dir)

        self.store(feature, idx)

def get_features_loader(dataset_path):
    data_loader = VideoIter(
        dataset_path=dataset_path,
        clip_length=16,
        frame_stride=1,
        video_transform=build_transforms(),
        return_label=False,
    )

    data_iter = torch.utils.data.DataLoader(
        data_loader,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    return data_loader, data_iter


if __name__ == "__main__":

    device = "cuda"
    dataset_path = r"C:\Users\macro\Desktop\prova" #sostituire con il path ai video da cui estrarre le features
    save_dir = "features"
    pretrained_3d = r".\FE\c3d.pickle"

    data_loader, data_iter = get_features_loader(dataset_path)

    network = load_feature_extractor(pretrained_3d, device).eval()

    if not path.exists(save_dir):
        mkdir(save_dir)

    features_writer = FeaturesWriter(num_videos=data_loader.video_count)
    with torch.no_grad():
        for data, clip_idxs, dirs, vid_names in data_iter:
            outputs = network(data.to(device)).detach().cpu().numpy()
            for i, (_dir, vid_name, clip_idx) in enumerate(zip(dirs, vid_names, clip_idxs)):
                _dir = path.join(save_dir, _dir)
                features_writer.write(
                    feature=outputs[i], video_name=vid_name, idx=clip_idx, dir=_dir,
                )

    features_writer.dump()
