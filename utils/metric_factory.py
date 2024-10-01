from sklearn import metrics
import math
from sklearn.metrics import roc_auc_score, roc_curve, balanced_accuracy_score, precision_recall_curve, auc, f1_score
from sklearn.metrics import average_precision_score
import numpy as np


def fast_bin_dice(actual, predicted):
    actual = np.asarray(actual).astype(bool) # bools are way faster to deal with by numpy
    predicted = np.asarray(predicted).astype(bool)
    im_sum = actual.sum() + predicted.sum()
    if im_sum == 0: return 1
    intersection = np.logical_and(actual, predicted)
    return 2. * intersection.sum() / im_sum

