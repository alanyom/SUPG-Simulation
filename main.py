import sys
sys.path.append("/Users/alanyom/Desktop") 

import pandas as pd
from supg.proxies.sentiment import SentimentProxy
from supg.sampling.adaptive import AdaptiveSampler
from supg.sampling.uniform import UniformSampler
from supg.constraints.targets import ConstraintChecker
from supg.evaluation.metrics import ThresholdOptimizer
from supg.config import DATA_DIR, ORACLE_BUDGET, LABEL_COL, PROXY_SCORE_COL

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def run_pipeline(input_csv: str, output_csv: str) -> None:
    df = pd.read_csv(DATA_DIR / input_csv)
    df = df[['labels', 'title', 'review']]
    
    # Step 1: proxy scoring
    proxy = SentimentProxy()
    df = proxy.score_batch(df, text_col="review")

    df_proxy = df.copy()
    
    # Step 2: adaptive sampling
    #sampler = AdaptiveSampler() 
    sampler = UniformSampler()
    df_sample = sampler.sample(df, n=ORACLE_BUDGET)

    
    # Step 3: oracle labeling
    # for unlabelled datasets, stop here
    # in this case, to test for convergence, im using a labeled dataset so this is unnecessary
    
    # Step 4: check constraints
    checker = ConstraintChecker()
    z1 = df_sample[df_sample[LABEL_COL] == 1].shape[0]
    z2 = 0  # simplified; replace with your findz() logic
    if checker.meets_targets(z1, z2, fp=5):  # Replace fp with actual FP count
        print("Targets met!")
    
    # Step 5: optimize threshold
    optimizer = ThresholdOptimizer()
    best_tau = optimizer.find_best_tau(df_sample)
    print(f"Best Tau: {optimizer.best_tau:.2f}")
    print(f"F1 Score: {optimizer.best_f1:.4f}")
    print(f"Precision: {optimizer.best_precision:.4f}")
    print(f"Recall: {optimizer.best_recall:.4f}")
    print(f"Upper Bound: {optimizer.best_upperbound:.4f}")
    print(f"Lower Bound: {optimizer.best_lowerbound:.4f}")

    # this is for convergence test with the labeled dataset
    print(df_sample[PROXY_SCORE_COL].describe())
    df_oracle = pd.read_csv(DATA_DIR / input_csv)
    df_proxy[LABEL_COL] = (df_proxy[PROXY_SCORE_COL] > best_tau).astype(int)
    df_oracle = df_oracle.reset_index(drop=True)
    df_proxy = df_proxy.reset_index(drop=True)

    # Compare the label columns
    accuracy = accuracy_score(df_oracle[LABEL_COL], df_proxy[LABEL_COL])
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Confusion Matrix: {confusion_matrix(df_oracle[LABEL_COL], df_proxy[LABEL_COL])}")

    
    df_sample.to_csv(DATA_DIR / output_csv, index=False)
    

if __name__ == "__main__":
    run_pipeline(input_csv="amazon_review_trial4.csv", output_csv="amazon_reviews_uniform_trial4.csv")