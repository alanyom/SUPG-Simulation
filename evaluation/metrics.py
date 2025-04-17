import numpy as np
import pandas as pd
from scipy.stats import norm
from supg.config import PROXY_SCORE_COL, LABEL_COL

class ThresholdOptimizer:
    def __init__(self):
        self.best_tau = 0.5
        self.best_f1 = -1
        self.best_precision = 0
        self.best_recall = 0
        self.best_upperbound = None
        self.best_lowerbound = None

    def find_best_tau(self, df: pd.DataFrame) -> float:
        # finding best tau
        
        for tau in np.arange(0, 1.01, 0.01):
            f1, precision, recall, upperbound, lowerbound = self._compute_metrics(df, tau)
            
            if f1 > self.best_f1:
                self.best_f1 = f1
                self.best_tau = tau
                self.best_precision = precision
                self.best_recall = recall
                self.best_upperbound = upperbound
                self.best_lowerbound = lowerbound

        return self.best_tau

    def _compute_metrics(self, df: pd.DataFrame, tau: float):
        tp = fp = fn = 0.0
        z1 = z2 = 0.0  
        
        for i in range(len(df)):
            score = df[PROXY_SCORE_COL].iloc[i]
            label = df[LABEL_COL].iloc[i]
            m = 1 / df['weight'].iloc[i] / len(df)  
            
            # calculate tp, fp, fn for precision/recall
            if score >= tau:
                if label == 1:
                    tp += m
                else:
                    fp += m
            else:
                if label == 1:
                    fn += m

            # calculate z1 and z2 for the confidence interval
            if score >= tau and label == 1:
                z1 += m
            elif score < tau and label == 1:
                z2 += m
        
        # precision, recall, and F1 score
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # compute confidence interval for z1, z2
        upperbound, lowerbound = self._confidence_interval(z1, z2)
        
        return f1, precision, recall, upperbound, lowerbound

    def _confidence_interval(self, z1, z2, cl=0.95):
        total_weight = z1 + z2
        if total_weight == 0:
            return np.nan, np.nan

        mean_z1 = z1 / total_weight
        mean_z2 = z2 / total_weight

        sd_z1 = np.sqrt(mean_z1 * (1 - mean_z1) / total_weight)
        sd_z2 = np.sqrt(mean_z2 * (1 - mean_z2) / total_weight)

        z_score = norm.ppf(1 - (1 - cl) / 2)

        upperbound = mean_z1 + z_score * sd_z1
        lowerbound = mean_z2 - z_score * sd_z2
        return upperbound, lowerbound
