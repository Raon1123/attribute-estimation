import numpy as np
from sklearn.metrics import average_precision_score, accuracy_score, f1_score


def mean_accuracy(preds, labels, eps=1e-10):
  """
  Calculate label-based evaluation metric mA

  Input
  - preds: (N, num_classes) np.array
  - labels: (N, num_classes) np.array

  Output
  - mA: (num_classes,)
  """
  num_classes = labels.shape[1]
  acc = np.zeros(num_classes)

  for i in range(num_classes):
    preds_i = preds[:, i] > 0.5
    labels_i = labels[:, i]

    # true positive
    tp = np.sum((preds_i == 1) & (labels_i == 1))
    # true negative
    tn = np.sum((preds_i == 0) & (labels_i == 0))
    # positive
    p = np.sum(labels_i == 1)
    # negative
    n = np.sum(labels_i == 0)

    acc[i] = tp / (p + eps) + tn / (n + eps)
    acc[i] = acc[i] / 2

  return acc


def example_based(preds, labels, eps=1e-10):
  """
  Calculate instance-based evaluation metric mA

  Input
  - preds: (N, num_classes)
  - labels: (N, num_classes)

  Output
  - Acc, Pre, Rec, F1 (N,)
  """
  num_instance = labels.shape[0]
  acc = np.zeros(num_instance)
  prec = np.zeros(num_instance)
  recall = np.zeros(num_instance)
  f1 = np.zeros(num_instance)

  for i in range(num_instance):
    preds_i = preds[i] > 0.5
    labels_i = labels[i]

    # true positive
    tp = np.sum((preds_i == 1) & (labels_i == 1))
    # true negative
    tn = np.sum((preds_i == 0) & (labels_i == 0))
    # false positive
    fp = np.sum((preds_i == 1) & (labels_i == 0))
    # false negative
    fn = np.sum((preds_i == 0) & (labels_i == 1))

    acc[i] = (tp + tn) / (tp + tn + fp + fn + eps)
    
    prec[i] = tp / (tp + fp + eps)
    recall[i] = tp / (tp + fn + eps)
    f1[i] = 2 * prec[i] * recall[i] / (prec[i] + recall[i] + eps)

  return acc, prec, recall, f1