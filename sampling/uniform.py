import pandas as pd

class UniformSampler:
    def __init__(self):
        pass 

    def sample(self, df: pd.DataFrame, n: int) -> pd.DataFrame:
        if len(df) <= n:
            return df.copy()

        df['weight'] = 1 / len(df)
        return df.sample(n=n, weights='weight').reset_index(drop=True)