import torch
from dataloader import Dataloader
from train import TorchModel
from network.anomaly_detector_model import AnomalyDetector
from loss import RegularizedLoss


features_path="features\c3d" #path alle features
annotation_path="Train_Annotation.txt" #annotazioni giuste

if __name__ == "__main__":

    device = "cuda"
    train_loader = Dataloader(features_path=features_path, annotation_path=annotation_path)
    network = AnomalyDetector(4096)
    model = TorchModel(network)
    model = model.to(device).train()
    optimizer = torch.optim.Adadelta(model.parameters(), lr=0.01, eps=1e-8)
    criterion = RegularizedLoss(network).to(device)

    model.fit(train_iter=train_loader, criterion=criterion, optimizer=optimizer, epochs=40)