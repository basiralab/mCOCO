# multi_sensory_memory_capacity.py

import numpy as np
import torch
import random
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from torchvision import datasets, transforms
from torch.nn.functional import interpolate
import librosa

from echoes import ESNRegressor

########################################
# Utility Functions
########################################

def shift_array(arr, n):
    shifted_arr = np.roll(arr, n)
    shifted_arr[:n] = 0
    return shifted_arr

def gen_lag_data(time_series, max_lag):
    x = time_series
    y = np.zeros((len(time_series), max_lag))
    for lag in range(1, max_lag + 1):
        y[:, lag - 1] = shift_array(x, lag)
    return x, y

def gen_lag_data_embeddings(embeddings, num_words, max_lag, seed=None):
    if seed is not None:
        random.seed(seed)
    indices = random.sample(range(len(embeddings)), num_words)
    selected_embeddings = embeddings[indices]
    x = selected_embeddings.flatten()
    y = np.zeros((num_words, max_lag))
    for lag in range(1, max_lag + 1):
        y[:, lag - 1] = shift_array(x, lag)
    return x, y

def generate_lagged_targets(data, max_lag):
    num_samples, input_dim = data.shape
    lagged = np.zeros((num_samples, max_lag, input_dim))
    for lag in range(1, max_lag + 1):
        lagged[lag:, lag - 1, :] = data[:-lag]
    return lagged.reshape(num_samples, -1)

def crop_and_resize(images, target_size=10):
    cropped_images = []
    for img in images:
        img_np = img.numpy()
        non_zero_rows = np.where(img_np.sum(axis=1) > 0)[0]
        non_zero_cols = np.where(img_np.sum(axis=0) > 0)[0]
        if len(non_zero_rows) > 0 and len(non_zero_cols) > 0:
            top, bottom = non_zero_rows[0], non_zero_rows[-1] + 1
            left, right = non_zero_cols[0], non_zero_cols[-1] + 1
            cropped = img_np[top:bottom, left:right]
            cropped_tensor = torch.tensor(cropped).unsqueeze(0).unsqueeze(0)
            resized = interpolate(cropped_tensor, size=(target_size, target_size), mode='bilinear', align_corners=False)
            cropped_images.append(resized.squeeze(0).squeeze(0))
        else:
            cropped_images.append(torch.zeros(target_size, target_size))
    return torch.stack(cropped_images)

def compute_memory_capacity(connectivity_matrix, X_train, y_train, X_test, y_test, max_lag):
    total_memory_capacity = 0
    esn = ESNRegressor(
        spectral_radius=0.99,
        input_scaling=1,
        leak_rate=1,
        bias=0,
        W=connectivity_matrix,
        random_state=42,
    )
    esn.fit(X_train, y_train)
    y_pred = esn.predict(X_test)
    for i in range(max_lag):
        true_values = y_test[:, i]
        predicted_values = y_pred[:, i]
        r, _ = pearsonr(true_values, predicted_values)
        r_squared = 0 if np.isnan(r) else r ** 2
        total_memory_capacity += r_squared
    return total_memory_capacity

def preprocess_audio(audio_signal, sr):
    mfcc = librosa.feature.mfcc(y=audio_signal, sr=sr, n_mfcc=13)
    return mfcc.T

def load_audio(file_path):
    return librosa.load(file_path, sr=None)

########################################
# MNIST Memory Capacity
########################################

def compute_mnist_memory_capacity(cbt_list):
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    num_samples, max_lag = 120, 20
    images = mnist_train.data[:num_samples].float() / 255.0
    images = crop_and_resize(images, target_size=15)
    mnist_x = images.view(num_samples, -1).numpy().astype(np.float32)
    mnist_y = generate_lagged_targets(mnist_x, max_lag)
    train_split = int(0.8 * num_samples)
    X_train, X_test = mnist_x[:train_split], mnist_x[train_split:]
    y_train, y_test = mnist_y[:train_split], mnist_y[train_split:]
    for cbt in cbt_list:
        conn_matrix = cbt.astype(np.float32)
        conn_matrix /= max(abs(np.linalg.eigvals(conn_matrix)))
        mc = compute_memory_capacity(conn_matrix, X_train, y_train, X_test, y_test, max_lag)
        print(f"MNIST Memory Capacity: {mc}")

########################################
# Gutenberg Embedding Memory Capacity
########################################

def compute_embedding_memory_capacity(cbt_list, embeddings_path, max_lag=20):
    embeddings = np.load(embeddings_path)
    X_train, y_train = gen_lag_data_embeddings(embeddings, 1000, max_lag, 41)
    X_test, y_test = gen_lag_data_embeddings(embeddings, 200, max_lag, 42)
    X_train = torch.from_numpy(X_train).unsqueeze(1).numpy()
    X_test = torch.from_numpy(X_test).unsqueeze(1).numpy()
    y_train = torch.from_numpy(y_train).numpy()
    y_test = torch.from_numpy(y_test).numpy()

    for cbt in cbt_list:
        conn_matrix = cbt.astype(np.float32)
        conn_matrix /= max(abs(np.linalg.eigvals(conn_matrix)))
        mc = compute_memory_capacity(conn_matrix, X_train.astype(np.float32), y_train.astype(np.float32),
                                     X_test.astype(np.float32), y_test.astype(np.float32), max_lag)
        print(f"Embedding Memory Capacity: {mc}")

########################################
# Audio Signal Memory Capacity
########################################

def compute_audio_memory_capacity(cbt_list, audio_file):
    audio_signal, sr = librosa.load(audio_file,sr = None)
    mfcc = preprocess_audio(audio_signal, sr)
    max_lag = 20
    X, y = gen_lag_data(mfcc[:, 0], max_lag)
    X = X.reshape(-1, 1)
    split = int(0.8 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]
    

    for cbt in cbt_list:
        conn_matrix = cbt.astype(np.float32)
        conn_matrix /= max(abs(np.linalg.eigvals(conn_matrix)))
        mc = compute_memory_capacity(conn_matrix, X_train, y_train, X_test, y_test, max_lag)
        print(f"Audio Memory Capacity: {mc}")

########################################
# Entry Point
########################################

if __name__ == '__main__':
    # Provide your own connectivity matrix lists here
    asd_results = np.load('/Users/mayssasoussia/Desktop/PhD/untitled folder 2/rc_asd_results.npy',allow_pickle=True)
    cbts = [fold['CBT'] for fold in asd_results]
    compute_mnist_memory_capacity(cbts)
    compute_embedding_memory_capacity(cbts, 'https://github.com/basiralab/mCOCO/blob/main/sensory_inputs/gutenberg_embeddings.npy.npy')
    compute_audio_memory_capacity(cbts,'https://github.com/basiralab/mCOCO/blob/main/sensory_inputs/quranic_recitation.mp3')
