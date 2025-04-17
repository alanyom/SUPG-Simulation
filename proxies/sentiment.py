from transformers import pipeline
import pandas as pd
from supg.config import PROXY_SCORE_COL

class SentimentProxy:
    def __init__(self):
        self.model = pipeline(
            model="lxyuan/distilbert-base-multilingual-cased-sentiments-student",
            return_all_scores=True,
            truncation=True
        )

    def score(self, text: str) -> float:
        # returns sentiment value
        result = self.model(text[:512]) 
        score = result[0][0]['score']
        print(score)
        return score

    def score_batch(self, df: pd.DataFrame, text_col: str) -> pd.DataFrame:
        df[PROXY_SCORE_COL] = df[text_col].apply(self.score)
        return df