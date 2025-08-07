

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

def compute_reservoir_states_keep_all(bold_signal, n_reservoir=111, alpha=0.5, spectral_radius=1.45, seed=42):
    np.random.seed(seed)
    W = np.random.uniform(-1, 1, (n_reservoir, n_reservoir))
    W_in = np.random.uniform(-1, 1, (n_reservoir, 1))
    eigvals = np.linalg.eigvals(W)
    W *= spectral_radius / max(abs(eigvals))

    n_timepoints, n_rois = bold_signal.shape
    reservoir_states = np.zeros((n_rois, n_timepoints))

    for roi in range(n_rois):
        x_roi = bold_signal[:, roi].reshape(-1, 1)
        h_t = np.zeros((n_reservoir, 1))
        h_t_updates = []
        for t in range(x_roi.shape[0]):
            h_t = (1 - alpha) * h_t + alpha * np.tanh(W_in * x_roi[t] + W @ h_t)
            h_t_updates.append(h_t)
        sums_updates = np.sum(h_t_updates, axis=1)
        reservoir_states[roi] = sums_updates.flatten()
    return reservoir_states

def run_cross_validation(asd_subjects, n_splits=5):
    roi_count = 111
    assert all(subject["ROI_Values"].shape[1] == roi_count for subject in asd_subjects.values())

    asd_ids = list(asd_subjects.keys())
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(asd_ids)):
        print(f"Processing Fold {fold + 1}")
        train_ids = [asd_ids[i] for i in train_idx]
        test_ids = [asd_ids[i] for i in test_idx]

        train_similarities = []
        for subject_id in train_ids:
            roi_values = asd_subjects[subject_id]["ROI_Values"]
            reservoir_states = compute_reservoir_states_keep_all(roi_values)
            similarity_matrix = np.corrcoef(reservoir_states)
            similarity_matrix[similarity_matrix < 0] = 0
            similarity_matrix[np.isnan(similarity_matrix)] = 0
            train_similarities.append(similarity_matrix)

        CBT_train = np.mean(train_similarities, axis=0)
        plt.figure()
        plt.imshow(CBT_train, cmap='viridis')
        plt.colorbar()
        plt.title(f"CBT - Fold {fold + 1}")
        plt.xlabel("ROIs")
        plt.ylabel("ROIs")
        plt.show()

        frobenius_distances = []
        test_similarities = []
        for subject_id in test_ids:
            roi_values = asd_subjects[subject_id]["ROI_Values"]
            reservoir_states = compute_reservoir_states_keep_all(roi_values)
            similarity_matrix = np.corrcoef(reservoir_states)
            similarity_matrix[similarity_matrix < 0] = 0
            similarity_matrix[np.isnan(similarity_matrix)] = 0
            test_similarities.append(similarity_matrix)
            distance = np.linalg.norm(CBT_train - similarity_matrix, ord='fro')
            frobenius_distances.append(distance)

        fold_results.append({
            "Fold": fold + 1,
            "Mean Frobenius Distance": np.mean(frobenius_distances),
            "Distances": frobenius_distances,
            "CBT": CBT_train,
            "test data": test_similarities
        })

    return fold_results

def plot_fold_summary(fold_results):
    overall_mean_fd = np.mean([f["Mean Frobenius Distance"] for f in fold_results])
    fold_numbers = [f["Fold"] for f in fold_results]
    mean_fds = [f["Mean Frobenius Distance"] for f in fold_results]

    print("\nCross-Validation Results:")
    for f in fold_results:
        print(f"Fold {f['Fold']}: Mean Frobenius Distance = {f['Mean Frobenius Distance']:.4f}")
    print(f"\nOverall Mean Frobenius Distance: {overall_mean_fd:.4f}")

    plt.figure(figsize=(8, 6))
    plt.bar(fold_numbers, mean_fds, color='blue', alpha=0.7)
    plt.axhline(overall_mean_fd, color='red', linestyle='--', label=f"Overall Mean ({overall_mean_fd:.4f})")
    plt.title("Mean Frobenius Distances Across Folds")
    plt.xlabel("Fold Number")
    plt.ylabel("Mean Frobenius Distance")
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def main():
    subjects_info = np.load('https://github.com/basiralab/mCOCO/blob/main/dataset/ABIDE_subset.npy',allow_pickle=True).item()

    asd_subjects = {k: v for k, v in subjects_info.items() if v["Group"] == 1}
    fold_results = run_cross_validation(asd_subjects)
    plot_fold_summary(fold_results)

if __name__ == "__main__":
    main()
