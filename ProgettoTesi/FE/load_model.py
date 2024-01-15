from train import TorchModel
from FE.c3d import C3D



def load_feature_extractor(feature_extractor_path, device):
    model = C3D(pretrained=feature_extractor_path)
    return model.to(device)


def load_anomaly_detector(ad_model_path, device):
    anomaly_detector = TorchModel.load_m(model_path=ad_model_path).to(device)
    return anomaly_detector.eval()


def load_models(
    feature_extractor_path,
    ad_model_path,
    device ="cuda"):

    feature_extractor = load_feature_extractor(feature_extractor_path, device)
    anomaly_detector = load_anomaly_detector(ad_model_path, device)
    return anomaly_detector, feature_extractor
