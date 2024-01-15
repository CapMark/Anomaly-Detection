import os
from torch import nn
import torch
from dataloader2 import DataloaderVal
import matplotlib.pyplot as plt


device = "cuda"


def load_model(model_path):
    model = torch.load(model_path, map_location="cpu")
    return model

""""plot validation or train loss"""
@staticmethod
def plot(trloss, n):
    it=[i for i in range (40)]
    plt.figure()
    plt.ylim([0, 1.5])
    plt.plot(it, trloss, color="navy")
    plt.xlabel("epoch")
    plt.ylabel(n)
    plt.savefig("results/"+n+".jpg")
    plt.close()

class TorchModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.device = "cuda"
        self.iteration = 0
        self.model = model

    @classmethod
    def load_m(cls, model_path):
        model = cls(load_model(model_path))
        return model



    def fit(self, train_iter, criterion, optimizer, epochs):
        path="results\model2"
        criterion = criterion.to(self.device)
        tloss=[]
        vloss=[]
        for epoch in range(epochs):
            total_loss = 0
            self.train()
            for iteration, (batch, targets) in enumerate(train_iter):
                self.iteration += 1
                batch = batch.to(device)
                targets = targets.to(device)
                outputs = self.model(batch)
                loss = criterion(outputs, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                self.iteration += 1
            val_loss = self.evaluate(criterion)
            torch.save(self.model, os.path.join(path, ("epoch_" + str(epoch + 1) + ".pt")))
            print("Epoch ", epoch + 1, "/", epochs, "loss ", total_loss/len(train_iter), "val loss:", val_loss)
            tloss.append(total_loss/len(train_iter))
            vloss.append(val_loss)



    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)




    def evaluate(self, criterion):
        features_path = "features\c3d 2" #sostituire con path per c3d features
        data_loader = DataloaderVal(features_path, annotation_path="Test2_Annotation.txt")
        data_iter = torch.utils.data.DataLoader(data_loader, batch_size=1, shuffle=False, num_workers=0,
                                                pin_memory=True)
        batches = torch.FloatTensor(500, 32, 4096) #SE SI TESTA SUL DATASET 1 SOSTITUIRE 500 CON 280
        targets = torch.FloatTensor(500)           #SE SI TESTA SUL DATASET 1 SOSTITUIRE 500 CON 280
        i=0
        self.eval()
        with torch.no_grad():

            for batch, start_end_couples, length in data_iter:
                c= start_end_couples[0]
                c=c[0]
                if c[0]==-1:
                    target=torch.zeros(1)
                else:
                    target=torch.ones(1)
                batches[i]=batch
                targets[i]=target
                i=i+1


            batch = batches.to(device)
            targets = targets.to(device)
            outputs = self.model(batch)
            loss = criterion(outputs, targets)
            loss = loss.item()
        return loss



