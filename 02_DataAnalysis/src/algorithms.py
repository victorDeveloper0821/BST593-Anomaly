import pandas as pd
import os
from typing import Union
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.svm import OneClassSVM

def find_best_k_bootstrap(df: pd.DataFrame, 
                max_k: int = 10,
                n_bootstrap=5,
                sample_size=8000,
                random_state=42,
                pca_reduce: bool = False, 
                pca_components: int = 2, 
                output: str = '../output/',
                Activity_type: str = 'Running'):
    """
Determine the optimal value of parameter k.

Parameters
----------
df : pandas.DataFrame
    Input dataset.
max_k : int
    Maximum value of k to evaluate.
n_bootstrap : int
    Number of bootstrap samples to generate.
sample_size : int
    Number of rows to include in each bootstrap sample.
pca_reduce : bool
    Whether to apply PCA for dimensionality reduction.
pca_components : int
    Number of PCA components to retain when PCA is enabled.
output : str or Path
    Directory in which to save generated plots.
activity_type : str
    Activity type to be analyzed.

Returns
-------
k : int
    The optimal value of k.
summary_df : pandas.DataFrame
    A summary DataFrame containing evaluation metrics for each k.
"""

    np.random.seed(random_state)
    numeric_df = df.select_dtypes(include=[np.number])
    data = numeric_df.to_numpy()

    if len(data) < sample_size:
        sample_size = len(data)

    silhouette_results = {k: [] for k in range(2, max_k + 1)}
    inertia_results = {k: [] for k in range(2, max_k + 1)}

    for b in range(n_bootstrap):
        idx = np.random.choice(len(data), size=sample_size, replace=False)
        subset = data[idx]

        # PCA dimension reduction
        if pca_reduce:
            pca = PCA(n_components=pca_components, random_state=random_state)
            subset = pca.fit_transform(subset)

        for k in range(2, max_k + 1):
            kmeans = MiniBatchKMeans(n_clusters=k, random_state=random_state, n_init=5, batch_size=600)
            labels = kmeans.fit_predict(subset)

            # silhouette score
            score = silhouette_score(subset, labels)
            inertia = kmeans.inertia_
            silhouette_results[k].append(score)
            inertia_results[k].append(inertia)

    # calculate mean and standard deviation for scores
    mean_scores = {k: np.mean(v) for k, v in silhouette_results.items()}
    std_scores = {k: np.std(v) for k, v in silhouette_results.items()}
    summary=[]
    for k in range(2, max_k+1): 
        score_mean = np.mean(silhouette_results[k]) 
        score_sem = np.std(silhouette_results[k]) / np.sqrt(len(silhouette_results[k]))
        upper = score_mean + 1.96*score_sem
        lower = score_mean - 1.96*score_sem

        summary.append({
            "k": k,
            "silhouette_mean": score_mean,
            "silhouette_lower": lower,
            "silhouette_upper": upper,
            "inertia_mean": np.mean(inertia_results[k])
        })

    summary_df = pd.DataFrame(summary)
    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    # Visualization
    # left axis Silhouette
    ax1.errorbar(
        summary_df["k"],
        summary_df["silhouette_mean"],
        yerr=[summary_df["silhouette_mean"] - summary_df["silhouette_lower"],
              summary_df["silhouette_upper"] - summary_df["silhouette_mean"]],
        fmt="o-", color="tab:blue", capsize=4, label="Silhouette (mean ± 95% CI)"
    )
    ax1.set_xlabel("Number of Clusters (k)", fontsize=11)
    ax1.set_ylabel("Silhouette Score", color="tab:blue", fontsize=11)
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    # right axis：Inertia
    ax2 = ax1.twinx()
    ax2.plot(
        summary_df["k"],
        summary_df["inertia_mean"],
        "s--", color="tab:orange", alpha=0.8, label="Inertia"
    )
    ax2.set_ylabel("Inertia", color="tab:orange", fontsize=11)
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    plt.title(f"Bootstrap-Averaged Silhouette and Inertia vs. k in {Activity_type}", fontsize=13)
    fig.tight_layout()
    ax1.grid(alpha=0.3)
    fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.9))
    filename = f'kmeans_parameter_{Activity_type}.png'
    plt.savefig(os.path.join(output, filename))
    plt.show()

    best_k = int(summary_df.loc[summary_df["silhouette_mean"].idxmax(), "k"])
    
    print(f"✅ Best k = {best_k}, mean silhouette ={summary_df['silhouette_mean'].max():.4f}")

    return best_k, summary_df

def kmeans_anomaly_detection(
    df: pd.DataFrame,
    unique_key: str = "unique_log_id",
    best_k: int = 2,
    pca_reduce: bool = False,
    pca_components: int = 2,
    confidence: float = 0.95,
    random_state: int = 42,
    batch_size: int = 4096,
    return_with_input: bool = True,
):
    """
MiniBatchKMeans-based anomaly detection.

Parameters
----------
df : pd.DataFrame
    Standardized input dataframe.
unique_key : str
    Column name representing the unique log identifier.
best_k : int
    Optimal number of clusters (k).
pca_reduce : bool
    Whether to apply PCA for dimensionality reduction.
pca_components : int
    Number of principal components to retain (default: 2).
confidence : float
    Confidence interval threshold (default: 0.95).
random_state : int
    Random seed for reproducibility.
batch_size : int
    Batch size used to mitigate memory constraints during MiniBatchKMeans fitting.
return_with_input : bool
    If True, return the anomaly scores merged with the original dataframe.

Returns
-------
pd.DataFrame or tuple
    If `return_with_input` is True, returns a DataFrame containing the original
    data merged with anomaly scores. Otherwise, returns a tuple containing:
    (results_df, lower_threshold, upper_threshold).
"""


    # --- Step 1: Select numeric columns only ---
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        raise ValueError("DataFrame must have numerical columns")

    data = numeric_df.to_numpy(copy=False) 

    # --- Step 2: Optional PCA (for speed-up or visualization) ---
    if pca_reduce:
        pca = PCA(n_components=pca_components, random_state=random_state)
        data = pca.fit_transform(data)
        print(f"PCA enabled → reduced to {pca_components} dimensions")

    # --- Step 3: MiniBatchKMeans clustering ---
    kmeans = MiniBatchKMeans(
        n_clusters=best_k,
        random_state=random_state,
        batch_size=batch_size,
        n_init=5,
        max_no_improvement=10,
        reassignment_ratio=0.02
    )
    labels = kmeans.fit_predict(data)
    centroids = kmeans.cluster_centers_

    # --- Step 4: Compute distances (vectorized) ---
    diffs = data - centroids[labels]
    distances = np.sqrt(np.einsum("ij,ij->i", diffs, diffs))

    # --- Step 5: Confidence interval (robust to large n) ---
    lower, upper = np.quantile(distances, [(1 - confidence) / 2, (1 + confidence) / 2])
    print(f"{int(confidence * 100)}% CI for centroid distances: ({lower:.4f}, {upper:.4f})")

    # --- Step 6: Mark anomalies ---
    percentile_threshold = 95
    threshold_distance = np.percentile(distances, percentile_threshold)
    anomalies = distances > threshold_distance

    # --- Step 7: Combine results ---
    result_df = pd.DataFrame({
        unique_key: df[unique_key].values,
        "cluster": labels.astype(np.int32),
        "distance_to_centroid": distances.astype(np.float32),
        "anomaly": anomalies.astype(bool),
    })

    if pca_reduce: 
        for i in range(pca_components):
            result_df[f"PCA{i+1}"] = data[:, i].astype(np.float32)

    # --- Step 8: Merge results back to input (memory-efficient join) ---
    if return_with_input:
        result_df.set_index(unique_key, inplace=True)
        merged_df = df.join(result_df, on=unique_key, how="left")
        return merged_df, float(lower), float(upper)
    else:
        return result_df, float(lower), float(upper)

def vs_kmeans_anomalies(result_df: pd.DataFrame, Activity_type: str, output: str): 
    """
Visualize anomaly detection results using a scatter plot.

Parameters
----------
result_df : pandas.DataFrame
    DataFrame containing the anomaly detection results.
output : str or Path
    Directory where the generated plot will be saved.
activity_type : str
    Activity type used for labeling or categorization in the plot.

Returns
-------
None
    """
    if not {"PCA1", "PCA2"}.issubset(result_df.columns):
        raise ValueError("PCA1, PCA2 columns are missing in df")
    plt.figure(figsize=(8, 6))

    # normal
    normal_df = result_df[~result_df["anomaly"]]
    plt.scatter(
        normal_df["PCA1"], normal_df["PCA2"],
        c=normal_df["cluster"], cmap="tab10", alpha=0.5, s=15, label="Normal"
    )

    # anomalies
    anomaly_df = result_df[result_df["anomaly"]]
    plt.scatter(
        anomaly_df["PCA1"], anomaly_df["PCA2"],
        c="red", s=30, label="Anomaly", edgecolor="black"
    )
    plt.title(f"KMeans PCA-based Anomaly Visualization in {Activity_type}", fontsize=13)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.savefig(os.path.join(output, f'kmeans_anomaly_{Activity_type}.png'))

    plt.show()
    
def oneClassSVM_anomaly_detection(
    df: pd.DataFrame,
    unique_key: str = "unique_log_id",
    subsample: int = 20000,
    nu: float = 0.01,
    gamma: str = "scale",
    random_state: int = 42,
    return_with_input: bool = True
):
    """
One-Class SVM–based anomaly detection with scaling and bootstrap sampling.

Parameters
----------
df : pd.DataFrame
    Standardized input dataframe.
unique_key : str
    Column name that uniquely identifies each log entry.
subsample : int
    Number of rows to sample for training the One-Class SVM.
nu : float
    An upper bound on the fraction of training errors and a lower bound
    on the fraction of support vectors.
gamma : str
    Kernel coefficient for the RBF kernel. Default is "scale".
return_with_input : bool
    If True, the resulting anomaly scores are merged with the original dataframe.

Returns
-------
tuple
    A tuple containing:
    - pd.DataFrame: DataFrame of anomaly scores (merged with input if specified).
    - OneClassSVM: The trained One-Class SVM model.
"""
    # Step 1: numerical columns
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        raise ValueError("DataFrame 沒有數值欄位可供 OneClassSVM 使用。")

    data = numeric_df.to_numpy(copy=False)

    # Step 2: bootstrap sampling
    n_samples = len(data)
    if n_samples > subsample:
        rng = np.random.default_rng(random_state)
        sample_idx = rng.choice(n_samples, subsample, replace=False)
        train_data = data[sample_idx]
        print(f"Using subsample of {subsample:,} rows for OneClassSVM training (from {n_samples:,})")
    else:
        train_data = data

    # Step 3: training model with rbf kernel (required scaled numerical values)
    ocsvm = OneClassSVM(kernel="rbf", nu=nu, gamma=gamma)
    ocsvm.fit(train_data)

    # Step 4: predict
    preds = ocsvm.predict(data)
    anomalies = preds == -1
    scores = ocsvm.decision_function(data) 

    # Step 5: return results
    result_df = pd.DataFrame({
        unique_key: df[unique_key].values,
        "anomaly": anomalies,
        "svm_score": scores
    })

    if return_with_input:
        merged_df = df.join(result_df.set_index(unique_key), on=unique_key, how="left")
        print(result_df["anomaly"].value_counts())
        return merged_df, ocsvm
    else:
        print(result_df["anomaly"].value_counts())
        return result_df, ocsvm

def vs_oneClassSVM_results(df: pd.DataFrame, 
    model: OneClassSVM,
    unique_key: str = "unique_log_id",
    anomaly_col: str = "anomaly",
    score_col: str = "svm_score",
    pca_components: int = 2,
    plot_boundary: bool = True,
    random_state: int = 42, 
    output: str='./output',
    Activity_type: str = 'Running'): 
    """
Visualize One-Class SVM anomaly detection results.

Parameters
----------
df : pandas.DataFrame
    Input DataFrame containing features and anomaly-related columns.
model : OneClassSVM
    Trained One-Class SVM model used for anomaly detection.
unique_key : str
    Column name representing the unique log identifier.
anomaly_col : str
    Column indicating the anomaly flag.
score_col : str
    Column containing the anomaly scores produced by the One-Class SVM model.
pca_components : int
    Number of PCA components used for dimensionality reduction.
plot_boundary : bool
    Whether to visualize the decision boundary of the One-Class SVM.
random_state : int
    Random seed used for reproducibility when applying PCA.
output : str or Path
    Directory where output plots will be saved.
Activity_type : str
    Activity type used for filtering or labeling the visualization.

Returns
-------
None
"""

    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty: 
        raise ValueError('Invalid dataframe')

    data = numeric_df.to_numpy(copy=False)
    anomalies = df[anomaly_col].values.astype(bool)

    ## dimension reduction
    pca = PCA(n_components=pca_components, random_state=random_state)
    data_2d = pca.fit_transform(data)
    print(f"PCA reduced data to {pca_components}D for visualization")
    if plot_boundary and pca_components == 2:
        ocsvm_vis = OneClassSVM(kernel="rbf", nu=model.nu, gamma=model.gamma)
        ocsvm_vis.fit(data_2d)

        x_min, x_max = data_2d[:, 0].min() - 0.1, data_2d[:, 0].max() + 0.1
        y_min, y_max = data_2d[:, 1].min() - 0.1, data_2d[:, 1].max() + 0.1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 400),
                             np.linspace(y_min, y_max, 400))
        grid = np.c_[xx.ravel(), yy.ravel()]
        Z = ocsvm_vis.decision_function(grid)
        Z = Z.reshape(xx.shape)

        # decision boundary
        plt.figure(figsize=(8, 6))
        plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 10),
                     cmap=plt.cm.PuBu, alpha=0.7)
        plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors="darkred")

        plt.scatter(
            data_2d[~anomalies, 0], data_2d[~anomalies, 1],
            s=10, c="tab:blue", label="Normal", alpha=0.6
        )
        plt.scatter(
            data_2d[anomalies, 0], data_2d[anomalies, 1],
            s=20, c="tab:red", label="Anomaly", alpha=0.8
        )

        plt.title(f"One-Class SVM Anomaly Detection (PCA 2D Space) in {Activity_type}")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.legend(loc="upper right")
        plt.grid(alpha=0.3)
        plt.tight_layout()

    else:
        print("⚠️ PCA components > 2: only scatter plot available (no boundary).")
        plt.figure(figsize=(8, 6))
        plt.scatter(
            data_2d[~anomalies, 0], data_2d[~anomalies, 1],
            s=10, c="tab:blue", label="Normal", alpha=0.6
        )
        plt.scatter(
            data_2d[anomalies, 0], data_2d[anomalies, 1],
            s=20, c="tab:red", label="Anomaly", alpha=0.8
        )
        plt.title(f"One-Class SVM Scatter (PCA Space) in {Activity_type}")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
    plt.savefig(os.path.join(output, f'oneClassSVM_{Activity_type}.png'))
    plt.show()
    
    
    