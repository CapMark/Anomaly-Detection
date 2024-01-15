from FE.load_model import load_anomaly_detector
import torch
from dataloader import read_features
import matplotlib.pyplot as plt

import os
import numpy as np

def ad_prediction(model, features, device="cuda"):
    features = torch.tensor(features).to(device)
    with torch.no_grad():
        preds = model(features)
    return preds.detach().cpu().numpy().flatten()


def plot(y_pred, v, len):
    x=int(len)/32
    pred=np.repeat(y_pred,x)
    seg=[i for i in range (int(len))]
    diff= seg.__len__()-pred.__len__()
    for e in range (diff):
        pred=np.append(pred, 0)
    plt.figure()
    plt.ylim([0, 1.1])
    plt.plot(seg, pred, color="navy")
    plt.xlabel("Frame")
    plt.ylabel("Prediction")
    plt.savefig(v+"jpg")
    plt.close()

def readAnnotation(v, ann='Test_Annotation.txt'):
    l=[]
    print(v)
    file = open(ann, 'r')
    for line in file:
        if v in line:
            l=line.split()
    file.close()
    return l[1]




if __name__ == "__main__":
    dataset=0 #1=UCF crime 0=real life violence situations
    anomaly_detector=load_anomaly_detector(ad_model_path="results/model/epoch_40.pt", device=torch.device("cuda"))
    l=os.listdir("features/VideoProva")
    for video in l:
        features=read_features("features/VideoProva/"+video, 4096)
        y_pred = ad_prediction(
            model=anomaly_detector,
            features=features,
        )
        if dataset==1:
            ann='Test_Annotation.txt'
            len = readAnnotation(video.replace(".txt", ""), ann)
        else:
            ann='Test2_Annotation.txt'
            len = readAnnotation(video.replace(".txt", ".mp4"), ann)


        plot(y_pred, "results/predictions/" + video.replace(".txt", ""), len)



