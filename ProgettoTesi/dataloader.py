import os
import numpy as np
import torch
from os import path
from torch.utils import data

class Dataloader:

    def __init__(self, features_path, annotation_path):
        super().__init__()
        self._features_path = features_path
        self.features_list_normal, self.features_list_anomaly = Dataloader._get_features_list(features_path=self._features_path, annotation_path=annotation_path)
        self._iterations = 1000
        self._batch_size = 30 #batchsize=50 se dataset 2 batchsize=30 se dataset 1
        self.feature_dim = 4096
        self._features_cache = {}
        self._i = 0

    def __len__(self):
        return self._iterations

    def __getitem__(self, index):
        if self._i == len(self):
            self._i = 0
            raise StopIteration

        feature, label = self.get_features()
        self._i += 1
        return feature, label

    def get_features(self):
        normal_paths = np.random.choice(self.features_list_normal, size=self._batch_size)
        abnormal_paths = np.random.choice(self.features_list_anomaly, size=self._batch_size)
        all_paths = np.concatenate([normal_paths, abnormal_paths])
        features = torch.stack([read_features(f"{feature_subpath}.txt", self.feature_dim, self._features_cache) for feature_subpath in all_paths])
        label=torch.cat([torch.zeros(self._batch_size), torch.ones(self._batch_size)])
        return features, label


    def _get_features_list(features_path: str, annotation_path: str):
        features_list_normal = []
        features_list_anomaly = []
        with open(annotation_path, "r") as f:
            lines = f.read().splitlines()
            for line in lines:
                items = line.split()
                file = items[0].split(".")[0]
                feature_path = os.path.join(features_path, file)
                if "Normal" in feature_path:
                    features_list_normal.append(feature_path)
                else:
                    features_list_anomaly.append(feature_path)
        return features_list_normal, features_list_anomaly


class DataloaderVal(data.Dataset):

    def __init__(self, features_path, annotation_path):
        super().__init__()
        self.features_path = features_path
        self.feature_dim = 4096
        self.features_list = DataloaderVal._get_features_list(features_path=features_path, annotation_path=annotation_path)

    def __len__(self):
        return len(self.features_list)

    def __getitem__(self, index: int):
        data = self.get_feature(index)
        return data

    def get_feature(self, index: int):
        feature_subpath, start_end_couples, length = self.features_list[index]
        features = read_features(f"{feature_subpath}.txt", self.feature_dim)
        return features, start_end_couples, length

    @staticmethod
    def _get_features_list(features_path: str, annotation_path: str):
        features_list = []
        with open(annotation_path, "r") as f:
            lines = f.read().splitlines()
            for line in lines:
                start_end_couples = []
                items = line.split()
                anomalies_frames = [int(x) for x in items[3:]]
                start_end_couples.append([anomalies_frames[0], anomalies_frames[1]])
                start_end_couples.append([anomalies_frames[2], anomalies_frames[3]])
                start_end_couples = torch.from_numpy(np.array(start_end_couples))
                file = items[0].split(".")[0]
                feature_path = os.path.join(features_path, file)
                length = int(items[1])
                features_list.append((feature_path, start_end_couples, length))
        return features_list

def read_features(file_path, feature_dim, cache=None):

    if cache is not None and file_path in cache:
        return cache[file_path]

    if not path.exists(file_path):
        raise FileNotFoundError(f"Feature doesn't exist: `{file_path}`")

    with open(file_path, "r") as fp:
        data = fp.read().splitlines()
        features = np.zeros((len(data), feature_dim))
        for i, line in enumerate(data):
            features[i, :] = [float(x) for x in line.split(" ")]

    features = torch.from_numpy(features).float()
    if cache is not None:
        cache[file_path] = features
    return features