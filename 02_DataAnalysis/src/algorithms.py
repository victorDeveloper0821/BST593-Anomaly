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
    自動找最佳 k
    回傳最佳 k（Silhouette Score 最大）
    可選擇 PCA 降維
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

        # PCA 選項
        if pca_reduce:
            pca = PCA(n_components=pca_components, random_state=random_state)
            subset = pca.fit_transform(subset)

        for k in range(2, max_k + 1):
            kmeans = MiniBatchKMeans(n_clusters=k, random_state=random_state, n_init=5, batch_size=600)
            labels = kmeans.fit_predict(subset)

            # silhouette 估計
            score = silhouette_score(subset, labels)
            inertia = kmeans.inertia_
            silhouette_results[k].append(score)
            inertia_results[k].append(inertia)

    # 計算每個 k 的平均 silhouette 分數
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
    
    # 視覺化
    # 左軸：Silhouette
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

    # 右軸：Inertia
    ax2 = ax1.twinx()
    ax2.plot(
        summary_df["k"],
        summary_df["inertia_mean"],
        "s--", color="tab:orange", alpha=0.8, label="Inertia"
    )
    ax2.set_ylabel("Inertia", color="tab:orange", fontsize=11)
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    # 標題與樣式
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
    MiniBatchKMeans-based anomaly detection
    ---------------------------------------
    適合大型 DataFrame（6萬筆以上資料），使用信賴區間判斷離群值。
    已針對 CPU/記憶體最佳化。

    Parameters
    ----------
    df : pd.DataFrame
        已正規化的輸入資料。
    unique_key : str
        每筆紀錄的唯一識別欄。
    best_k : int
        已事先選定的最佳 K 值。
    pca_reduce : bool
        是否使用 PCA 降維（僅影響可視化與速度，不影響距離計算）。
    pca_components : int
        PCA 降維維度（預設 2）。
    confidence : float
        信賴區間（預設 0.95）。
    random_state : int
        隨機種子以保持重現性。
    batch_size : int
        MiniBatchKMeans 的 batch 大小，影響速度與記憶體。
    return_with_input : bool
        若 True，將結果合併回原始 DataFrame。

    Returns
    -------
    pd.DataFrame or (pd.DataFrame, float, float)
        標註 cluster、距離與 anomaly 標記後的 DataFrame。
        同時回傳信賴區間上下界 (lower, upper)。
    """

    # --- Step 1: Select numeric columns only ---
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        raise ValueError("DataFrame 沒有數值欄位可供 KMeans 分群。")

    data = numeric_df.to_numpy(copy=False)  # 避免記憶體複製

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
    distances = np.sqrt(np.einsum("ij,ij->i", diffs, diffs))  # 高效距離運算

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
    """ Visualized Anomaly detection in scatter plot"""
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
    One-Class SVM 異常偵測（效能優化版）
    - 不進行 scaling（假設輸入已正規化）
    - 自動抽樣訓練，避免記憶體爆掉
    - 支援高維度與大樣本資料
    """

    # Step 1: 僅取數值欄位
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        raise ValueError("DataFrame 沒有數值欄位可供 OneClassSVM 使用。")

    data = numeric_df.to_numpy(copy=False)  # 不複製記憶體

    # Step 2: 抽樣訓練子集（防止 kernel memory 爆掉）
    n_samples = len(data)
    if n_samples > subsample:
        rng = np.random.default_rng(random_state)
        sample_idx = rng.choice(n_samples, subsample, replace=False)
        train_data = data[sample_idx]
        print(f"Using subsample of {subsample:,} rows for OneClassSVM training (from {n_samples:,})")
    else:
        train_data = data

    # Step 3: 建立與訓練模型
    ocsvm = OneClassSVM(kernel="rbf", nu=nu, gamma=gamma)
    ocsvm.fit(train_data)

    # Step 4: 預測 (1=正常, -1=異常)
    preds = ocsvm.predict(data)
    anomalies = preds == -1
    scores = ocsvm.decision_function(data)  # 越小越異常

    # Step 5: 回傳結果
    result_df = pd.DataFrame({
        unique_key: df[unique_key].values,
        "anomaly": anomalies,
        "svm_score": scores
    })

    if return_with_input:
        # join 比 merge 記憶體效率更好
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
        # 在 PCA 空間重新擬合模型以畫 decision boundary
        ocsvm_vis = OneClassSVM(kernel="rbf", nu=model.nu, gamma=model.gamma)
        ocsvm_vis.fit(data_2d)

        x_min, x_max = data_2d[:, 0].min() - 0.1, data_2d[:, 0].max() + 0.1
        y_min, y_max = data_2d[:, 1].min() - 0.1, data_2d[:, 1].max() + 0.1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 400),
                             np.linspace(y_min, y_max, 400))
        grid = np.c_[xx.ravel(), yy.ravel()]
        Z = ocsvm_vis.decision_function(grid)
        Z = Z.reshape(xx.shape)

        # 畫 decision boundary
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
    
    
    