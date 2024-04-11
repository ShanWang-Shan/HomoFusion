import torch

from torchmetrics import Metric
from typing import List, Optional
from .data.apolloscape.trainId2color import labels
Evaluate_label = [not l.ignoreInEval for l in labels]

class BaseIoUMetric(Metric):
    """
    Computes intersection over union at given thresholds
    """
    def __init__(self, classes, thresholds=[0.4, 0.3, 0.2]):
        super().__init__(dist_sync_on_step=False, compute_on_step=False)

        if classes == 1:
            thresholds = torch.FloatTensor(thresholds)
            self.add_state('thresholds', default=thresholds, dist_reduce_fx='mean')
            self.add_state('tp', default=torch.zeros(len(thresholds)), dist_reduce_fx='sum')
            self.add_state('fp', default=torch.zeros(len(thresholds)), dist_reduce_fx='sum')
            self.add_state('fn', default=torch.zeros(len(thresholds)), dist_reduce_fx='sum')
        else:
            self.add_state('tp', default=torch.zeros(classes), dist_reduce_fx='sum')
            self.add_state('fp', default=torch.zeros(classes), dist_reduce_fx='sum')
            self.add_state('fn', default=torch.zeros(classes), dist_reduce_fx='sum')

        self.classes = classes


    def update(self, pred, label):
        if self.classes == 1: # waterhazard
            pred = pred.detach().sigmoid().reshape(-1)
            label = label.detach().bool().reshape(-1)

            pred = pred[:, None] >= self.thresholds[None]
            label = label[:, None]

            self.tp += (pred & label).sum(0)
            self.fp += (pred & ~label).sum(0)
            self.fn += (~pred & label).sum(0)
        elif self.classes == 3: # waymo:
            pred = pred.detach().sigmoid()
            label = label.detach().bool()
            max_pred,_ = torch.max(pred, dim=1)
            pred = (pred == max_pred[:,None])
            pred = pred.permute(1, 0, 2, 3) # c b h w
            label = label.permute(1, 0, 2, 3) # c b h w

            pred = pred.flatten(1)  # c p
            label = label.flatten(1)  # c p

            self.tp += (pred & label).sum(1)
            self.fp += (pred & ~label).sum(1)
            self.fn += (~pred & label).sum(1)
        else: # apolloscape
            pred = pred.detach().sigmoid()
            label = label.detach().bool()
            max_pred,_ = torch.max(pred, dim=1)
            pred = (pred == max_pred[:,None])
            pred = pred.permute(1, 0, 2, 3) # c b h w
            label = label.permute(1, 0, 2, 3) # c b h w

            # do not calculate ignore part
            if label.shape[0] > self.tp.shape[0]:
                pred &= ~label[-1]
                pred &= ~label[-2]
                label = label[:self.tp.shape[0]] & ~label[-1]
                label &= ~label[-2]
            pred = pred.flatten(1) # c p
            label = label.flatten(1) # c p

            self.tp += (pred & label).sum(1)
            self.fp += (pred & ~label).sum(1)
            self.fn += (~pred & label).sum(1)

    def compute(self):
        if self.classes == 1:
            thresholds = self.thresholds.squeeze(0)
            ious = self.tp / (self.tp + self.fp + self.fn + 1e-7)
            precision = self.tp / (self.tp + self.fp + 1e-7)
            recall = self.tp / (self.tp + self.fn + 1e-7)
            f1score = (2 * precision * recall) / (precision + recall)

            rtn = {}
            for t, i ,p, r, f in zip(thresholds, ious, precision, recall, f1score):
                sub = {}
                sub['iou'] = i.item()
                sub['precision'] = p.item()
                sub['recall'] = r.item()
                sub['fiscore'] = f.item()
                rtn[f'@{t.item():.2f}'] = sub
            return rtn
        else:
            ious = self.tp / (self.tp + self.fp + self.fn + 1e-7)
            rtn = {}
            all_ious = torch.mean(ious, dim=0)
            rtn['@all']  = {'iou': all_ious.item()}

            unignore_ious = ious[Evaluate_label[:ious.shape[0]]]
            unignore = torch.mean(unignore_ious, dim=0)
            rtn['@unignore'] = {'iou': unignore.item()}

            return rtn

class IoUMetric(BaseIoUMetric):
    def __init__(self, classes, label_indices: List[List[int]], min_visibility: Optional[int] = None):
        """
        label_indices:
            transforms labels (c, h, w) to (len(labels), h, w)
            see config/experiment/* for examples

        min_visibility:
            passing "None" will ignore the visibility mask
            otherwise uses visibility values to ignore certain labels
            visibility mask is in order of "increasingly visible" {1, 2, 3, 4, 255 (default)}
            see https://github.com/nutonomy/nuscenes-devkit/blob/master/docs/schema_nuscenes.md#visibility
        """
        super().__init__(classes)

        self.label_indices = label_indices
        self.min_visibility = min_visibility

    def update(self, pred, batch):
        if 'bev' in pred.keys():
            if isinstance(pred, dict):
                pred = pred['bev']                                                              # b c h w

            label = batch['bev']                                                                # b n h w
            label = [label[:, idx].max(1, keepdim=True).values for idx in self.label_indices]
            label = torch.cat(label, 1)                                                         # b c h w
        else:
            pred = pred['mask']  # b c h w
            label = batch['mask']  # b c h w

        if self.min_visibility is not None:
            mask = batch['visibility'] >= self.min_visibility
            mask = mask[:, None].expand_as(pred)                                            # b c h w

            pred = pred[mask]                                                               # m
            label = label[mask]                                                             # m

        return super().update(pred, label)
