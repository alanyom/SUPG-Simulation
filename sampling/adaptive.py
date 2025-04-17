import numpy as np
import pandas as pd

class AdaptiveSampler:
    def __init__(self):
        pass

    def sample(self, df: pd.DataFrame, n: int) -> pd.DataFrame:
        if len(df) <= n:
            return df.copy()

        df['sqrt_proxy'] = np.sqrt(df['proxy_score'])
        # importance sampling and normalization
        df['weight'] = (df['sqrt_proxy'] * 0.9) + (0.1 * 1/len(df))
        df['weight'] = df['weight'] / df['weight'].sum()
        return df.sample(n=n, weights='weight')