import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import savgol_filter
import scalib.attacks as attacks
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr, kendalltau
import time
import pandas as pd
from joblib import Parallel, delayed

waveforms = np.load("waveforms.npy")
inputs = np.load("inputs.npy")
weights = np.load("weights.npy")

num_traces, num_samples = waveforms.shape
num_candidates = 10000
apply_scaling = True
apply_filtering = True
window_size = 11

if apply_scaling:
    waveforms = StandardScaler().fit_transform(waveforms)

if apply_filtering:
    waveforms = savgol_filter(waveforms, window_length=window_size, polyorder=2, axis=1)

candidates = np.linspace(-2, 2, num_candidates)
correlation_matrix = np.zeros((2, num_candidates))
spearman_matrix = np.zeros((2, num_candidates))
kendall_matrix = np.zeros((2, num_candidates))

cpa_function = getattr(attacks, "cpa", None)

def compute_correlations(weight_idx):
    start_time = time.time()
    leakage_hypotheses = np.array([inputs[:, weight_idx] * w for w in candidates]).T
    cpa_result = cpa_function(waveforms, leakage_hypotheses)
    max_corr = np.max(np.abs(cpa_result), axis=0)
    spearman_corr = [spearmanr(leakage_hypotheses[:, i], waveforms[:, i])[0] for i in range(num_candidates)]
    kendall_corr = [kendalltau(leakage_hypotheses[:, i], waveforms[:, i])[0] for i in range(num_candidates)]
    best_weight = candidates[np.argmax(max_corr)]
    execution_time = time.time() - start_time
    return weight_idx, best_weight, max_corr, spearman_corr, kendall_corr, execution_time

if cpa_function:
    results = Parallel(n_jobs=2)(delayed(compute_correlations)(i) for i in range(2))
    recovered_weights = []
    
    for weight_idx, best_weight, max_corr, spearman_corr, kendall_corr, execution_time in results:
        correlation_matrix[weight_idx, :] = max_corr
        spearman_matrix[weight_idx, :] = spearman_corr
        kendall_matrix[weight_idx, :] = kendall_corr
        recovered_weights.append(best_weight)
        print(f"Weight {weight_idx} recovery time: {execution_time:.2f} seconds")
    
    print(f"Recovered Weights: {recovered_weights}")
    print(f"Actual Weights: {weights}")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(correlation_matrix, cmap="coolwarm", xticklabels=200, yticklabels=50)
    plt.title("CPA Correlation Heatmap (SCALib)")
    plt.xlabel("Candidate Weights")
    plt.ylabel("Weight Index (0=First Weight, 1=Second Weight)")
    plt.show()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(spearman_matrix, cmap="coolwarm", xticklabels=200, yticklabels=50)
    plt.title("Spearman Correlation Heatmap (SCALib)")
    plt.xlabel("Candidate Weights")
    plt.ylabel("Weight Index (0=First Weight, 1=Second Weight)")
    plt.show()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(kendall_matrix, cmap="coolwarm", xticklabels=200, yticklabels=50)
    plt.title("Kendall Tau Correlation Heatmap (SCALib)")
    plt.xlabel("Candidate Weights")
    plt.ylabel("Weight Index (0=First Weight, 1=Second Weight)")
    plt.show()
    
    pearson_corr = [pearsonr(candidates, correlation_matrix[i])[0] for i in range(2)]
    spearman_corr = [spearmanr(candidates, spearman_matrix[i])[0] for i in range(2)]
    kendall_corr = [kendalltau(candidates, kendall_matrix[i])[0] for i in range(2)]
    df_corr = pd.DataFrame({"Weight Index": [0, 1], "Pearson": pearson_corr, "Spearman": spearman_corr, "Kendall": kendall_corr})
    print(df_corr)
