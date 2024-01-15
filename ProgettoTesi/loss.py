import torch

def custom_objective(y_pred, y_true):
    lambdas = 0.00008
    normal_vids_indices = torch.where(y_true == 0)
    anomal_vids_indices = torch.where(y_true == 1)

    normal_segments_scores = y_pred[normal_vids_indices].squeeze(-1)
    anomal_segments_scores = y_pred[anomal_vids_indices].squeeze(-1)

    normal_segments_scores_maxes = normal_segments_scores.max(dim=-1)[0]
    anomal_segments_scores_maxes = anomal_segments_scores.max(dim=-1)[0]

    hinge_loss = 1 - anomal_segments_scores_maxes + normal_segments_scores_maxes
    hinge_loss = torch.max(hinge_loss, torch.zeros_like(hinge_loss))

    smoothed_scores = anomal_segments_scores[:, 1:] - anomal_segments_scores[:, :-1]
    smoothed_scores_sum_squared = smoothed_scores.pow(2).sum(dim=-1)

    sparsity_loss = anomal_segments_scores.sum(dim=-1)

    final_loss = (hinge_loss + lambdas * smoothed_scores_sum_squared + lambdas * sparsity_loss).mean()
    return final_loss


class RegularizedLoss(torch.nn.Module):

    def __init__(self, model, lambdas=0.001):
        super().__init__()
        self.lambdas = lambdas
        self.model = model


    def forward(self, y_pred, y_true):
        fc1_params = torch.cat(tuple([x.view(-1) for x in self.model.fc1.parameters()]))
        fc2_params = torch.cat(tuple([x.view(-1) for x in self.model.fc2.parameters()]))
        fc3_params = torch.cat(tuple([x.view(-1) for x in self.model.fc3.parameters()]))

        l1_regularization = self.lambdas * torch.norm(fc1_params, p=2)
        l2_regularization = self.lambdas * torch.norm(fc2_params, p=2)
        l3_regularization = self.lambdas * torch.norm(fc3_params, p=2)

        return (custom_objective(y_pred, y_true)+ l1_regularization+ l2_regularization+ l3_regularization)
