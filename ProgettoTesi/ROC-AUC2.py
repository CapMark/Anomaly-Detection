import torch
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
from train import load_model
from dataloader import DataloaderVal

device = "cuda"
features_path="features\c3d 2" #sostituire con il path alle features
model_Path="results\model\epoch_40.pt" #sostituire con il modello addestrato sul dataset GIUSTO
data_loader = DataloaderVal(features_path, annotation_path="Test2_Annotation.txt") #sostituire con le annotazioni giuste
data_iter = torch.utils.data.DataLoader(data_loader, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
model = load_model(model_path=model_Path).to(device).eval()


def plot(fpr, tpr, auc):
    plt.figure()
    auc=f'{auc:.1f}'
    plt.plot(fpr, tpr, color="orange",  label="ROC curve (area = "+str(auc) + ")" )
    plt.plot([0, 1], [0, 1], color="navy")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.savefig("results\ROC2.png")
    plt.close()

if __name__ == "__main__":
    y_trues = torch.tensor([])
    y_preds = torch.tensor([])
    nvideo=0

    with torch.no_grad():
        for features, intervals, vlen in data_iter:
            nvideo+=1
            print("Video n: "+ str(nvideo)+"/500") #280 se test 1 500 se test 2
            features = features.to(device)
            outputs = model(features).squeeze(-1)
            for vid_len, interval, output in zip(vlen, intervals, outputs.cpu().numpy()):
                y_true = np.zeros(vlen)
                y_pred = np.zeros(vlen)

                segments_len = torch.div(vid_len, 32,rounding_mode='floor')
                for couple in interval:
                    if couple[0] != -1:
                        y_true[couple[0] : couple[1]] = 1
                for i in range(32):
                    segment_start_frame = i * segments_len
                    segment_end_frame = (i + 1) * segments_len
                    y_pred[segment_start_frame:segment_end_frame] = output[i]
                if y_trues is None:
                    y_trues = y_true
                    y_preds = y_pred
                else:
                    y_trues = np.concatenate([y_trues, y_true])
                    y_preds = np.concatenate([y_preds, y_pred])
    fpr, tpr, thresholds = roc_curve(y_true=y_trues, y_score=y_preds, pos_label=1)
    auc = auc(fpr, tpr)
    plot(fpr, tpr, auc)
    print("AUC= ", auc)
