import os
import json
import gdown
import random

import numpy as np
from scipy.io import loadmat

from task_and_baseline import baseline, build_task_helpers


def set_seed(seed: int = 42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

set_seed(42)


# Download the dataset
if not os.path.exists('challenge.mat'):
    url = "https://drive.google.com/file/d/1BBHVSI4KB-B8OX46eN1Nm4ARCeq6Rui4/view?usp=sharing"
    downloaded_file = "challenge.mat"
    gdown.download(url, downloaded_file, quiet=False, fuzzy=True) # TypeError: download() got an unexpected keyword argument 'fuzzy'

data = loadmat("challenge.mat", simplify_cells=True)
tx = data["tx"].astype(np.complex128)
rx = data["rx"].astype(np.complex128)
Fs = float(data["Fs"])
N, _ = tx.shape

tx_n = tx / (np.sqrt(np.mean(np.abs(tx) ** 2, axis=0, keepdims=True)) + 1e-30)
helpers = build_task_helpers(tx_n, Fs, N)


def rank1_from_matrix(matrix):
    """
    Find 1-rank component in SVD.
    """
    cov = matrix.conj().T @ matrix / matrix.shape[0]
    _, vecs = np.linalg.eigh(cov)
    shared = matrix @ vecs[:, -1] # project matrix onto eigenvector corresponding to the maximum eigenvalue
    denom = np.vdot(shared, shared) + 1e-30
    # restore rank-1 component for all channels
    return np.column_stack([
        (np.vdot(shared, matrix[:, ch]) / denom) * shared
        for ch in range(matrix.shape[1])
    ])

def your_canceller(tx_n, rx):
    # 1. find tx nonlinearity (without E correction) from baseline
    tx_pred = helpers["fit_tx_prediction"](rx)
    rx_res_with_E = rx - tx_pred
    
    # 2. 1-rank source E correction with SVD
    E_pred = rank1_from_matrix(rx_res_with_E)
    
    rx_hat = rx_res_with_E - E_pred
    
    return rx_hat


print("\n=== Baseline ===")
baseline_reds, baseline_avg = helpers["score"](
    rx, baseline(tx_n, rx, helpers["fit_tx_prediction"]), label="baseline"
)

print("=== Your Solution ===")
yours_reds, yours_avg = helpers["score"](rx, your_canceller(tx_n, rx), label="yours")

results = {
    "baseline": {
        "per_channel_db": baseline_reds,
        "average_db": baseline_avg,
    },
    "yours": {
        "per_channel_db": yours_reds,
        "average_db": yours_avg,
    },
}

with open("results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)


#################################################
# Old solutions (all iterations: from 1 to 9)

# def rank1_from_band_matrix(band_matrix):
#     """
#     Function from baseline helpers. Delete E source.
#     """
#     cov = band_matrix.conj().T @ band_matrix / band_matrix.shape[0]
#     _, vecs = np.linalg.eigh(cov)
#     shared = band_matrix @ vecs[:, -1]
#     denom = np.vdot(shared, shared) + 1e-30
#     return np.column_stack([
#         (np.vdot(shared, band_matrix[:, ch]) / denom) * shared
#         for ch in range(band_matrix.shape[1])
#     ])

# def your_canceller_1(tx_n, rx):
#     # 1. tx nonlinearity (without E correction)
#     tx_pred = helpers["fit_tx_prediction"](rx)
#     rx_residual = rx - tx_pred
    
#     # 2. spatial residual correction
#     # use filter to leave only noise and E source
#     score_filter = helpers["score_filter"]
#     num_rx = rx.shape[1]
    
#     rx_residual_band = np.column_stack([
#         score_filter(rx_residual[:, ch]) for ch in range(num_rx)
#     ])
    
#     # use function from baseline
#     E_pred = rank1_from_band_matrix(rx_residual_band)
    
#     rx_hat = rx_residual - E_pred
    
#     return rx_hat

# def your_canceller_2(tx_n, rx):
#     score_filter = helpers["score_filter"]
#     N, num_rx = rx.shape
#     num_tx = tx_n.shape[1]
    
#     # 1. Make 72 model terms
#     model_terms = []
    
#     # linear
#     for i in range(num_tx):
#         model_terms.append(tx_n[:, i])
        
#     # cubic
#     for i in range(num_tx):
#         # self-distortion
#         model_terms.append(tx_n[:, i] * np.abs(tx_n[:, i])**2)
        
#         # cross-modulation
#         for j in range(num_tx):
#             if i != j:
#                 model_terms.append(tx_n[:, i]**2 * tx_n[:, j].conj())
#                 model_terms.append(tx_n[:, i] * np.abs(tx_n[:, j])**2)
                
#     filtered_terms = [score_filter(t) for t in model_terms]
    
#     for term in filtered_terms:
#         term.setflags(write=False)
    
#     # 2. fit tx prediction
#     MODEL_LAGS = tuple(range(-6, 7))
#     MODEL_SUBSET = slice(20_000, 520_000)
    
#     model_x = np.column_stack([
#         shifted_window(term, lag, MODEL_SUBSET.start, MODEL_SUBSET.stop)
#         for term in filtered_terms
#         for lag in MODEL_LAGS
#     ])
    
#     # regularization
#     model_gram = model_x.conj().T @ model_x + 1e-6 * np.eye(model_x.shape[1])
#     model_x.setflags(write=False)
#     model_gram.setflags(write=False)
    
#     pred = np.zeros_like(rx)
#     for ch in range(num_rx):
#         y = score_filter(rx[:, ch])[MODEL_SUBSET]
#         coef = np.linalg.solve(model_gram, model_x.conj().T @ y)
#         coef = coef.reshape(len(filtered_terms), len(MODEL_LAGS))
        
#         ch_pred = np.zeros(N, dtype=np.complex128)
#         for term_idx, term in enumerate(filtered_terms):
#             for lag_idx, lag in enumerate(MODEL_LAGS):
#                 if np.abs(coef[term_idx, lag_idx]) > 1e-10: # throw away low terms
#                     ch_pred += coef[term_idx, lag_idx] * shift_signal(term, lag)
                    
#         pred[:, ch] = ch_pred

#     rx_residual = rx - pred
    
#     # 3. E correction
#     rx_residual_band = np.column_stack([
#         score_filter(rx_residual[:, ch]) for ch in range(num_rx)
#     ])
    
#     # use function from baseline
#     E_pred = rank1_from_band_matrix(rx_residual_band)
    
#     rx_hat = rx_residual - E_pred
    
#     return rx_hat

# def your_canceller_3(tx_n, rx):
#     score_filter = helpers["score_filter"]
#     N, num_rx = rx.shape
#     num_tx = tx_n.shape[1]
    
#     MODEL_LAGS = tuple(range(-6, 7))
#     MODEL_SUBSET = slice(20_000, 520_000)
    
#     # Feature preparation: 13 lags * 6 channels * 2 (Real/Imag) = 156 features
#     X_train_list = []
#     X_all_list = []
    
#     for lag in MODEL_LAGS:
#         for i in range(num_tx):
#             term = tx_n[:, i]
            
#             shifted_train = shifted_window(term, lag, MODEL_SUBSET.start, MODEL_SUBSET.stop)
#             X_train_list.extend([shifted_train.real, shifted_train.imag])
            
#             shifted_full = shift_signal(term, lag)
#             X_all_list.extend([shifted_full.real, shifted_full.imag])
            
#     X_train = np.column_stack(X_train_list)
#     X_all = np.column_stack(X_all_list)
    
#     tx_pred = np.zeros_like(rx)
    
#     print(f"Training 8 models (4 channels * Real/Imag) on {X_train.shape[0]} rows...")
    
#     for ch in range(num_rx):
#         print(f"  -> Training for the RX channel {ch}...")
        
#         # The target variable (y) is the filtered error,
#         # since we only want to predict the interference in the desired band.
#         y_full = score_filter(rx[:, ch])
#         y_train = y_full[MODEL_SUBSET]
        
#         # training real part
#         reg_real = HistGradientBoostingRegressor(max_iter=100, random_state=42)
#         reg_real.fit(X_train, y_train.real)
#         pred_real = reg_real.predict(X_all)
        
#         # training imag part
#         reg_imag = HistGradientBoostingRegressor(max_iter=100, random_state=42)
#         reg_imag.fit(X_train, y_train.imag)
#         pred_imag = reg_imag.predict(X_all)
        
#         tx_pred[:, ch] = pred_real + 1j * pred_imag
        
#         # Smooth the tree response with a filter.
#         tx_pred[:, ch] = score_filter(tx_pred[:, ch])

#     rx_residual = rx - tx_pred
    
#     print("E correction...")
    
#     # 2. E correction
#     rx_residual_band = np.column_stack([
#         score_filter(rx_residual[:, ch]) for ch in range(num_rx)
#     ])
    
#     # use function from baseline
#     E_pred = rank1_from_band_matrix(rx_residual_band)
    
#     rx_hat = rx_residual - E_pred
    
#     return rx_hat

# def your_canceller_4(tx_n, rx):
#     score_filter = helpers["score_filter"]
#     num_rx = rx.shape[1]
    
#     def get_E_pred_raw(signal):
#         # 1. Apply filter to find the spatial signature of the interference
#         band = np.column_stack([score_filter(signal[:, ch]) for ch in range(num_rx)])
        
#         # 2. Compute covariance matrix (4x4) and principal eigenvector (v)
#         cov = band.conj().T @ band / band.shape[0]
#         _, vecs = np.linalg.eigh(cov)
#         v = vecs[:, -1]
        
#         # 3. Find projection coefficients for each channel
#         shared_band = band @ v
#         denom = np.vdot(shared_band, shared_band) + 1e-30
#         c = [(np.vdot(shared_band, band[:, ch]) / denom) for ch in range(num_rx)]
        
#         # 4. Apply vector and coefficients to the raw signal
#         shared_raw = signal @ v
#         E_raw = np.column_stack([c[ch] * shared_raw for ch in range(num_rx)])
        
#         return E_raw

#     print("  [Optimal] Запуск Joint Estimation (Alternating Projections)...")
    
#     # 1. Coarse estimation
#     tx_pred = helpers["fit_tx_prediction"](rx)
#     E_pred = get_E_pred_raw(rx - tx_pred)
    
#     # Refinement
#     tx_pred = helpers["fit_tx_prediction"](rx - E_pred)
#     E_pred = get_E_pred_raw(rx - tx_pred)
    
#     # Final refinement
#     tx_pred = helpers["fit_tx_prediction"](rx - E_pred)
#     E_pred = get_E_pred_raw(rx - tx_pred)
    
#     rx_hat = rx - tx_pred - E_pred
    
#     return rx_hat

# class SICNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # Вход: 12 каналов (6 TX * 2 Real/Imag)
#         # kernel_size=13 и padding=6 точно повторяют лаг от -6 до +6
#         self.net = nn.Sequential(
#             nn.Conv1d(in_channels=12, out_channels=64, kernel_size=13, padding=6),
#             nn.GELU(),
            
#             # MLP-блок: нелинейное смешивание признаков без сдвига во времени
#             nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1),
#             nn.GELU(),
            
#             # Выход: 8 каналов (4 RX * 2 Real/Imag)
#             nn.Conv1d(in_channels=64, out_channels=8, kernel_size=1)
#         )

#     def forward(self, x):
#         return self.net(x)

# def your_canceller_5(tx_n, rx):
#     score_filter = helpers["score_filter"]
#     N, num_rx = rx.shape
    
#     MODEL_SUBSET = slice(20_000, 220_000)
    
#     # 1. Выбор устройства для PyTorch
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"\n  [NN] Инициализация PyTorch (устройство: {device})...")

#     # 2. Подготовка данных (Перевод Complex -> Real/Imag)
#     tx_real = np.column_stack([tx_n.real, tx_n.imag])  # shape: (N, 12)
#     rx_real = np.column_stack([rx.real, rx.imag])      # shape: (N, 8)

#     # Формируем батч для обучения (Batch=1, Channels, Time)
#     X_train = torch.tensor(tx_real[MODEL_SUBSET], dtype=torch.float32).transpose(0, 1).unsqueeze(0).to(device)
#     Y_train = torch.tensor(rx_real[MODEL_SUBSET], dtype=torch.float32).transpose(0, 1).unsqueeze(0).to(device)

#     # 3. Инициализация модели и оптимизатора
#     model = SICNet().to(device)
#     optimizer = optim.Adam(model.parameters(), lr=0.01)
#     criterion = nn.MSELoss()

#     print(f"  [NN] Обучение нейросети на {X_train.shape[2]} сэмплах (Wideband MSE)...")
    
#     # 4. Обучение (150 эпох хватит для быстрой сходимости)
#     epochs = 150
#     model.train()
#     for epoch in range(epochs):
#         optimizer.zero_grad()
#         output = model(X_train)
#         loss = criterion(output, Y_train)
#         loss.backward()
#         optimizer.step()
        
#         if (epoch + 1) % 50 == 0:
#             print(f"       Эпоха {epoch+1}/{epochs} | Loss: {loss.item():.6f}")

#     # 5. Предсказание TX-помехи для ВСЕГО сигнала
#     print("  [NN] Предсказание помехи...")
#     model.eval()
#     with torch.no_grad():
#         X_all = torch.tensor(tx_real, dtype=torch.float32).transpose(0, 1).unsqueeze(0).to(device)
#         Y_pred_all = model(X_all).squeeze(0).transpose(0, 1).cpu().numpy()

#     # Склеиваем Real и Imag обратно в Complex
#     tx_pred_complex = np.zeros_like(rx)
#     for ch in range(num_rx):
#         # 0-3 каналы - это Real, 4-7 каналы - это Imag
#         tx_pred_complex[:, ch] = Y_pred_all[:, ch] + 1j * Y_pred_all[:, ch + num_rx]

#     # 6. Вычитаем TX-помеху (Wideband)
#     rx_residual = rx - tx_pred_complex
    
#     # 7. Извлекаем внешнюю помеху E (Narrowband SVD)
#     print("  [NN] Извлечение когерентной помехи (E)...")
#     rx_residual_band = np.column_stack([
#         score_filter(rx_residual[:, ch]) for ch in range(num_rx)
#     ])
#     E_pred = rank1_from_band_matrix(rx_residual_band)
    
#     # 8. Финальный результат
#     rx_hat = rx_residual - E_pred
    
#     return rx_hat

# 1. Пишем кастомную комплексную активацию
# class ComplexGELU(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # Используем обычный GELU "под капотом"
#         self.gelu = nn.GELU()

#     def forward(self, x):
#         # Применяем активацию независимо к амплитудам I и Q (Real/Imag)
#         return self.gelu(x.real) + 1j * self.gelu(x.imag)

# 2. Комплексная архитектура сети
# class ComplexSICNet(nn.Module):
#     def __init__(self, in_features=6, out_features=4, n_lags=13, dtype=torch.complex128):
#         super().__init__()
        
#         self.act = nn.Tanh()
#         self.conv1 = nn.Conv1d(in_channels=in_features, out_channels=64, kernel_size=n_lags, padding=int((n_lags-1)/2), dtype=dtype)
#         self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1, dtype=dtype)
#         self.conv3 = nn.Conv1d(in_channels=64, out_channels=out_features, kernel_size=1, dtype=dtype)

#     def forward(self, x):
#         x = self.act(self.conv1(x))
#         x = self.act(self.conv2(x))
#         return self.conv3(x)


# def your_canceller_6(tx_n, rx):
#     score_filter = helpers["score_filter"]
#     N, num_rx = rx.shape
#     MODEL_SUBSET = slice(20_000, 620_000)
#     DTYPE = torch.complex128
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     print(f"\n[Complex NN] Инициализация PyTorch (устройство: {device})...")

#     # 3. Подготовка данных теперь ЭЛЕМЕНТАРНАЯ
#     # Никакого разбиения! Передаем сырые комплексные массивы
#     X_train = torch.tensor(tx_n[MODEL_SUBSET], dtype=DTYPE).transpose(0, 1).unsqueeze(0).to(device)
#     Y_train = torch.tensor(rx[MODEL_SUBSET], dtype=DTYPE).transpose(0, 1).unsqueeze(0).to(device)

#     model = ComplexSICNet(dtype=DTYPE).to(device)
    
#     # 4. ВАЖНО: Обычный MSE работает некорректно для комплексных чисел, 
#     # так как ошибка должна вычисляться через сопряжение: |y - y_hat|^2
#     # PyTorch-оптимизаторы любят скаляры, поэтому мы явно считаем мощностную ошибку:
#     def complex_mse(pred, target):
#         err = pred - target
#         return torch.mean(err.real**2 + err.imag**2)
    
#     epochs = 200
#     warmap_ep = int(epochs*0.1)
#     start_lr = 1e-2
#     target_lr = 1e-6
    
#     optimizer = optim.AdamW(model.parameters(), lr=start_lr, weight_decay=1e-4)
    
#     scheduler = SequentialLR(optimizer, [
#         LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmap_ep),
#         CosineAnnealingLR(optimizer, T_max=epochs - warmap_ep, eta_min=target_lr)
#     ], milestones=[warmap_ep])

#     print(f"[Complex NN] Обучение сети ({X_train.shape[2]} сэмплов)...")
    
#     model.train()
#     for epoch in range(epochs):
#         optimizer.zero_grad()
#         output = model(X_train)
#         loss = complex_mse(output, Y_train)
#         loss.backward()
#         clip_utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#         optimizer.step()
#         scheduler.step()

#         print(f"  Эпоха {epoch+1}/{epochs} | Loss: {loss.item():.6f} | LR: {scheduler.get_last_lr()[0]:.2e}")

#     print("[Complex NN] Предсказание помехи...")
    
#     model.eval()
#     with torch.no_grad():
#         X_all = torch.tensor(tx_n, dtype=DTYPE).transpose(0, 1).unsqueeze(0).to(device)
#         tx_pred_complex = model(X_all).squeeze(0).transpose(0, 1).cpu().numpy()

#     # Вычитаем TX-помеху
#     rx_residual = rx - tx_pred_complex
    
#     print("[Complex NN] Удаление когерентной помехи E...")
#     rx_residual_band = np.column_stack([
#         score_filter(rx_residual[:, ch]) for ch in range(num_rx)
#     ])
#     E_pred = rank1_from_band_matrix(rx_residual_band)
    
#     rx_hat = rx_residual - E_pred
#     return rx_hat

# class ComplexTanh(nn.Module):
#     def forward(self, x):
#         return torch.tanh(x.real) + 1j * torch.tanh(x.imag)

# class CorrectionNet(nn.Module):
#     def __init__(self, in_features=6, out_features=4, n_lags=13, dtype=torch.complex128):
#         super().__init__()
        
#         # --- Branch 1: F_c(TX) ---
#         # no bias!
#         self.tx_net = nn.Sequential(
#             nn.Conv1d(in_features, 64, kernel_size=n_lags, padding=n_lags//2, bias=False, dtype=dtype),
#             ComplexTanh(),
#             nn.Conv1d(64, 64, kernel_size=1, bias=False, dtype=dtype),
#             ComplexTanh(),
#             nn.Conv1d(64, out_features, kernel_size=1, bias=False, dtype=dtype)
#         )
        
#         # --- Branch 2: E * f(t) ---
#         self.E_vec = nn.Parameter(torch.randn(out_features, dtype=dtype) * 0.1)
        
#         self.f_t_mlp = nn.Sequential(
#             nn.Conv1d(1, 16, kernel_size=1, dtype=dtype),
#             ComplexTanh(),
#             nn.Conv1d(16, 1, kernel_size=1, dtype=dtype)
#         )
        
#         # --- Branch 3: rx_hat(t) shift ---
#         self.rx_drift_mlp = nn.Sequential(
#             nn.Conv1d(1, 32, kernel_size=1, dtype=dtype),
#             ComplexTanh(),
#             nn.Conv1d(32, out_features, kernel_size=1, dtype=dtype)
#         )
        
#         # init wights
#         self.apply(self._init_weights)
        
#         # bias for f(t) to make it 1.0 at start
#         with torch.no_grad():
#             self.f_t_mlp[-1].bias.real.fill_(1.0)
#             self.f_t_mlp[-1].bias.imag.fill_(0.0)
    
#     def _init_weights(self, m):
#         """ custom Xavier init complex weights for Tanh """
#         if isinstance(m, nn.Conv1d):
#             # calculate fan_in и fan_out for conv
#             fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight)
            
#             # Coefficient Gain for Tanh is 5/3 (~1.666)
#             gain = nn.init.calculate_gain('tanh')
            
#             # Xavier std
#             std = gain * np.sqrt(2.0 / (fan_in + fan_out))
            
#             # for complex std devides by sqrt(2), 
#             # for Var(Real) + Var(Imag) = Var(Xavier)
#             std_cplx = std / np.sqrt(2.0)
            
#             with torch.no_grad():
#                 m.weight.real.normal_(0, std_cplx)
#                 m.weight.imag.normal_(0, std_cplx)
            
#             if m.bias is not None:
#                 nn.init.zeros_(m.bias)

#     def forward(self, tx, t_complex):
#         # 1. TX
#         fc_tx = self.tx_net(tx)
        
#         # 2. E * f(t)
#         f_t = self.f_t_mlp(t_complex)  # shape: (Batch, 1, Time)
#         E_complex = self.E_vec.view(1, -1, 1) # shape: (1, 4, 1)
#         E_f_t = E_complex * f_t # Broadcasting
        
#         # 3. rx_drift
#         rx_drift_t = self.rx_drift_mlp(t_complex)
        
#         # sum
#         rx_pred = fc_tx + E_f_t + rx_drift_t
        
#         return rx_pred, fc_tx, E_f_t, rx_drift_t


# def your_canceller_7(tx_n, rx):
#     MODEL_SUBSET = slice(20_000, 220_000)
#     DTYPE = torch.complex128
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     print(f"\nInit data (device: {device})...")

#     N = rx.shape[0]
    
#     # normilize t and make it complex
#     t_raw = np.linspace(-1.0, 1.0, N, dtype=np.float64)
#     t_complex_np = (t_raw + 1j * t_raw).reshape(N, 1)

#     # data to tensors (Batch=1, Channels, Time)
#     X_tx = torch.tensor(tx_n[MODEL_SUBSET], dtype=DTYPE).transpose(0, 1).unsqueeze(0).to(device)
#     X_t = torch.tensor(t_complex_np[MODEL_SUBSET], dtype=DTYPE).transpose(0, 1).unsqueeze(0).to(device)
#     Y_train = torch.tensor(rx[MODEL_SUBSET], dtype=DTYPE).transpose(0, 1).unsqueeze(0).to(device)

#     model = CorrectionNet(dtype=DTYPE).to(device)
    
#     # Adam stage
#     lr_start = 1e-2
#     lr_end = 1e-5
#     optimizer = optim.AdamW(model.parameters(), lr=lr_start, weight_decay=1e-4)
    
#     epochs_adam = 150
#     scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_adam, eta_min=lr_end)
    
#     def complex_mse(pred, target):
#         err = pred - target
#         return torch.mean(err.real**2 + err.imag**2)

#     print(f"Train joint-architecture ({X_tx.shape[2]} data samples)...")
    
#     print(f"[Phase 1] Adam train ({epochs_adam} epochs)...")
#     model.train()
#     for epoch in range(epochs_adam):
#         optimizer.zero_grad()
#         rx_pred, _, _, _ = model(X_tx, X_t)
        
#         loss = complex_mse(rx_pred, Y_train)
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.5)
#         optimizer.step()
#         scheduler.step()
        
#         if (epoch + 1) % 5 == 0:
#             current_lr = scheduler.get_last_lr()[0]
#             print(f"  Adam Epoch {epoch+1}/{epochs_adam} | LR: {current_lr:.3e} | MSE Loss: {loss.item():.3e}")
    
#     # L-BFGS stage
#     epochs_lbfgs = 15
    
#     optimizer_lbfgs = optim.LBFGS(
#         model.parameters(), 
#         lr=0.1,
#         max_iter=20, 
#         tolerance_grad=1e-7,
#         tolerance_change=1e-9, 
#         line_search_fn="strong_wolfe"
#     )
    
#     print(f"\n[Phase 2] L-BFGS train ({epochs_lbfgs} epochs)...")
#     for epoch in range(epochs_lbfgs):
        
#         # L-BFGS closure, to calculate MSE loss
#         def closure():
#             optimizer_lbfgs.zero_grad()
#             rx_pred, _, _, _ = model(X_tx, X_t)
#             loss = complex_mse(rx_pred, Y_train)
#             loss.backward()
#             return loss

#         optimizer_lbfgs.step(closure)
        
#         with torch.no_grad():
#             rx_pred, _, _, _ = model(X_tx, X_t)
#             current_loss = complex_mse(rx_pred, Y_train)
            
#         print(f"  L-BFGS Epoch {epoch+1}/{epochs_lbfgs} | MSE Loss: {current_loss.item():.3e}")

#     print("Correction calculating...")
    
#     model.eval()
#     with torch.no_grad():
#         X_tx_all = torch.tensor(tx_n, dtype=DTYPE).transpose(0, 1).unsqueeze(0).to(device)
#         X_t_all = torch.tensor(t_complex_np, dtype=DTYPE).transpose(0, 1).unsqueeze(0).to(device)
        
#         _, fc_tx_pred, E_ft_pred, _ = model(X_tx_all, X_t_all)
        
#         fc_tx_np = fc_tx_pred.squeeze(0).transpose(0, 1).cpu().numpy()
#         E_ft_np = E_ft_pred.squeeze(0).transpose(0, 1).cpu().numpy()

#     # substract correction
#     rx_hat = rx - fc_tx_np - E_ft_np
    
#     return rx_hat

# def your_canceller_8(tx_n, rx):
#     """
#     Physics-Informed Neural Network (Grey-Box Model) for Signal Interference Cancellation.
#     """
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"\nInitializing Grey-Box Model (device: {device})...")

#     # The subset of data used for training (to save time and memory)
#     MODEL_SUBSET = slice(20_000, 220_000)
#     DTYPE = torch.complex128
#     score_filter = helpers["score_filter"]
    
#     # -------------------------------------------------------------------------
#     # 1. FEATURE ENGINEERING: Generate the 10 authorized non-linear TX terms
#     # -------------------------------------------------------------------------
#     print("Generating and filtering mathematical TX features...")
#     tx_terms_np = np.column_stack([
#         tx_n[:, 0] ** 2 * tx_n[:, 1].conj(),
#         tx_n[:, 1] ** 2 * tx_n[:, 0].conj(),
#         tx_n[:, 0] ** 2 * tx_n[:, 3].conj(),
#         tx_n[:, 3] ** 2 * tx_n[:, 0].conj(),
#         tx_n[:, 1] ** 2 * tx_n[:, 2].conj(),
#         tx_n[:, 2] ** 2 * tx_n[:, 1].conj(),
#         tx_n[:, 3] ** 2 * tx_n[:, 2].conj(),
#         tx_n[:, 2] ** 2 * tx_n[:, 3].conj(),
#         tx_n[:, 0] ** 2 * tx_n[:, 5].conj(),
#         tx_n[:, 5] ** 2 * tx_n[:, 0].conj(),
#     ])

#     # Filter features and RX target to focus the network ONLY on the target frequency band.
#     # Learning wide-band noise is useless and causes overfitting.
#     tx_terms_filtered = np.zeros_like(tx_terms_np)
#     for i in range(10):
#         tx_terms_filtered[:, i] = score_filter(tx_terms_np[:, i])

#     rx_filtered = np.zeros_like(rx)
#     for i in range(4):
#         rx_filtered[:, i] = score_filter(rx[:, i])

#     # -------------------------------------------------------------------------
#     # 2. PREPARE PYTORCH TENSORS
#     # -------------------------------------------------------------------------
#     # Tensors for training (Batch=1, Channels, Time)
#     X_tx_train = torch.tensor(tx_terms_filtered[MODEL_SUBSET], dtype=DTYPE).T.unsqueeze(0).to(device)
#     Y_rx_train = torch.tensor(rx_filtered[MODEL_SUBSET], dtype=DTYPE).T.unsqueeze(0).to(device)

#     # -------------------------------------------------------------------------
#     # 3. DEFINE THE PHYSICS-INFORMED NEURAL NETWORK
#     # -------------------------------------------------------------------------
#     class PhysicsInformedNet(nn.Module):
#         def __init__(self, n_terms=10, n_rx=4, n_lags=13):
#             super().__init__()
#             # Branch 1: FIR filters for the TX terms. 
#             # STRICTLY NO BIAS and NO TANH to maintain 100% mathematical explainability!
#             self.tx_filters = nn.Conv1d(
#                 in_channels=n_terms, 
#                 out_channels=n_rx, 
#                 kernel_size=n_lags, 
#                 padding=n_lags // 2, 
#                 bias=False, 
#                 dtype=DTYPE
#             )
            
#             # Branch 2: Spatial signature of the external interference (Rank-1 constraint)
#             # This represents the physical location/phase of the external source E
#             self.spatial_vector = nn.Parameter(torch.ones((n_rx, 1), dtype=DTYPE))

#             # Small random initialization to break symmetry
#             with torch.no_grad():
#                 self.tx_filters.weight.real.normal_(0, 1e-4)
#                 self.tx_filters.weight.imag.normal_(0, 1e-4)

#         def forward(self, tx_terms, rx_target):
#             # 1. Predict TX non-linear interference (explainable subspace)
#             tx_pred = self.tx_filters(tx_terms)

#             # 2. Extract the Rank-1 Spatial Interference (E) dynamically
#             residual = rx_target - tx_pred
            
#             # Normalize spatial vector
#             w = self.spatial_vector
#             w_norm = w / (torch.norm(w) + 1e-12)
#             w_conj = w_norm.conj().view(1, -1, 1) # Shape: (1, 4, 1)

#             # Project residual onto the spatial vector to extract the shared time-domain signal
#             # This acts like a differentiable Principal Component Analysis (PCA) / SVD step
#             shared_time = torch.sum(residual * w_conj, dim=1, keepdim=True) # Shape: (Batch, 1, Time)

#             # Reconstruct the strictly Rank-1 interference across all 4 channels
#             e_pred = shared_time * w_norm.view(1, -1, 1)

#             return tx_pred, e_pred

#     model = PhysicsInformedNet().to(device)

#     def complex_mse(pred, target):
#         err = pred - target
#         return torch.mean(err.real**2 + err.imag**2)

#     # -------------------------------------------------------------------------
#     # 4. TRAINING LOOP
#     # -------------------------------------------------------------------------
#     print("Training Joint Architecture (TX FIR + Spatial E)...")
    
#     # Phase 1: Adam Optimizer for rapid initial convergence
#     optimizer = optim.AdamW(model.parameters(), lr=0.01)
#     epochs_adam = 150
    
#     model.train()
#     for epoch in range(epochs_adam):
#         optimizer.zero_grad()
#         tx_pred, e_pred = model(X_tx_train, Y_rx_train)
        
#         # We penalize the remaining unexplainable residual
#         loss = complex_mse(tx_pred + e_pred, Y_rx_train)
#         loss.backward()
#         optimizer.step()

#         if (epoch + 1) % 50 == 0:
#             print(f"  Adam Epoch {epoch+1}/{epochs_adam} | MSE Loss: {loss.item():.4e}")

#     # Phase 2: L-BFGS Optimizer for high-precision fine-tuning
#     lbfgs = optim.LBFGS(
#         model.parameters(), 
#         lr=0.1, 
#         max_iter=20, 
#         tolerance_grad=1e-9, 
#         tolerance_change=1e-12, 
#         line_search_fn="strong_wolfe"
#     )
    
#     epochs_lbfgs = 30
#     for epoch in range(epochs_lbfgs):
#         def closure():
#             lbfgs.zero_grad()
#             tx_p, e_p = model(X_tx_train, Y_rx_train)
#             loss = complex_mse(tx_p + e_p, Y_rx_train)
#             loss.backward()
#             return loss
            
#         lbfgs.step(closure)
#         if (epoch + 1) % 5 == 0:
#             print(f"  L-BFGS Epoch {epoch+1}/{epochs_lbfgs} | MSE Loss: {closure().item():.4e}")

#     # -------------------------------------------------------------------------
#     # 5. INFERENCE ON FULL DATASET
#     # -------------------------------------------------------------------------
#     print("Applying learned physical model to the full signal...")
#     model.eval()
#     with torch.no_grad():
#         X_tx_full = torch.tensor(tx_terms_filtered, dtype=DTYPE).T.unsqueeze(0).to(device)
#         Y_rx_full = torch.tensor(rx_filtered, dtype=DTYPE).T.unsqueeze(0).to(device)
        
#         tx_pred_full, e_pred_full = model(X_tx_full, Y_rx_full)
        
#         # Convert back to numpy (Time, Channels)
#         tx_np = tx_pred_full.squeeze(0).T.cpu().numpy()
#         e_np = e_pred_full.squeeze(0).T.cpu().numpy()

#     # Subtract the structured interference from the ORIGINAL RAW rx signal
#     rx_hat = rx - tx_np - e_np
    
#     return rx_hat

# print("\n=== Baseline ===")
# baseline_reds, baseline_avg = helpers["score"](
#     rx, baseline(tx_n, rx, helpers["fit_tx_prediction"]), label="baseline"
# )

# class DeepUnfoldedCanceller(nn.Module):
#     """
#     Physics-Informed Deep Unfolded Network.
#     Uses gradient descent to jointly optimize the TX subspace and Rank-1 Spatial Attention.
#     """
#     def __init__(self, tx_init, w_init, dtype=torch.complex128):
#         super().__init__()
#         self.tx_layer = nn.Linear(130, 4, bias=False, dtype=dtype)
#         # ИСПРАВЛЕНИЕ: Добавлен .contiguous() чтобы избежать ошибки L-BFGS
#         self.tx_layer.weight.data = torch.tensor(tx_init, dtype=dtype).T.contiguous()
        
#         self.spatial_layer = nn.Linear(1, 4, bias=False, dtype=dtype)
#         # ИСПРАВЛЕНИЕ: Добавлен .contiguous()
#         self.spatial_layer.weight.data = torch.tensor(w_init, dtype=dtype).contiguous()

#     def forward(self, tx_feat, target):
#         # 1. Predict TX interference
#         tx_pred = self.tx_layer(tx_feat)
        
#         # 2. Dynamic Differentiable Rank-1 Extraction
#         res = target - tx_pred
#         w = self.spatial_layer.weight
#         w_norm = w / (torch.norm(w) + 1e-12)
        
#         # Project residual onto spatial vector (Extract temporal signal E)
#         e_time = res @ w_norm.conj() 
#         # Reconstruct exactly Rank-1 spatial interference
#         e_pred = e_time @ w_norm.T
        
#         return tx_pred + e_pred

# # -------------------------------------------------------------------------
# # 2. Main Canceller Logic
# # -------------------------------------------------------------------------
# def your_canceller_9(tx_n, rx):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"\nInitializing Deep Unfolded Network (device: {device})...")

#     MODEL_LAGS = tuple(range(-6, 7))
#     MODEL_SUBSET = slice(20_000, 220_000)
#     DTYPE = torch.complex128
#     score_filter = helpers["score_filter"]

#     # --- Step A: Generate physics features & Initial Weights ---
#     print("Precomputing physics-informed features...")
    
#     def shift_signal(x, k):
#         y = np.zeros_like(x)
#         if k >= 0: y[k:] = x[:len(x)-k]
#         else:
#             kk = -k
#             y[:len(x)-kk] = x[kk:]
#         return y

#     model_terms = [
#         score_filter(tx_n[:, 0] ** 2 * tx_n[:, 1].conj()),
#         score_filter(tx_n[:, 1] ** 2 * tx_n[:, 0].conj()),
#         score_filter(tx_n[:, 0] ** 2 * tx_n[:, 3].conj()),
#         score_filter(tx_n[:, 3] ** 2 * tx_n[:, 0].conj()),
#         score_filter(tx_n[:, 1] ** 2 * tx_n[:, 2].conj()),
#         score_filter(tx_n[:, 2] ** 2 * tx_n[:, 1].conj()),
#         score_filter(tx_n[:, 3] ** 2 * tx_n[:, 2].conj()),
#         score_filter(tx_n[:, 2] ** 2 * tx_n[:, 3].conj()),
#         score_filter(tx_n[:, 0] ** 2 * tx_n[:, 5].conj()),
#         score_filter(tx_n[:, 5] ** 2 * tx_n[:, 0].conj()),
#     ]

#     model_x_np = np.column_stack([
#         shift_signal(term, lag)[MODEL_SUBSET]
#         for term in model_terms for lag in MODEL_LAGS
#     ])
#     model_gram_np = model_x_np.conj().T @ model_x_np + 1e-6 * np.eye(130)

#     # Filter target rx to train the NN only in the valid band
#     rx_filt_train = np.column_stack([score_filter(rx[:, c]) for c in range(4)])[MODEL_SUBSET]

#     # Smart Initialization (Warm-Start)
#     tx_init = np.zeros((130, 4), dtype=np.complex128)
#     for ch in range(4):
#         tx_init[:, ch] = np.linalg.solve(model_gram_np, model_x_np.conj().T @ rx_filt_train[:, ch])
        
#     tx_pred_train = model_x_np @ tx_init
#     res_train = rx_filt_train - tx_pred_train
#     cov = res_train.conj().T @ res_train
#     _, evecs = np.linalg.eigh(cov)
#     w_init = evecs[:, -1:] # Principal component

#     # --- Step B: Train the PyTorch Network ---
#     print("Training Joint Optimization Network via L-BFGS...")
#     model = DeepUnfoldedCanceller(tx_init, w_init, dtype=DTYPE).to(device)
    
#     # ИСПРАВЛЕНИЕ: Гарантируем contiguous() для входных данных
#     TX_pt = torch.tensor(model_x_np, dtype=DTYPE).contiguous().to(device)
#     Y_pt = torch.tensor(rx_filt_train, dtype=DTYPE).contiguous().to(device)

#     # L-BFGS will perfectly find the global joint optimum
#     optimizer = optim.LBFGS(model.parameters(), lr=1.0, max_iter=20, tolerance_grad=1e-7)
    
#     model.train()
#     for epoch in range(5):
#         def closure():
#             optimizer.zero_grad()
#             pred = model(TX_pt, Y_pt)
#             loss = torch.mean(torch.abs(Y_pt - pred)**2)
#             loss.backward()
#             return loss
#         optimizer.step(closure)
#         print(f"  Unfolded Net Epoch {epoch+1}/5 | MSE Loss: {closure().item():.4e}")

#     # --- Step C: Global Inference (Zero Hallucination) ---
#     print("Constructing global interference signal...")
#     model.eval()
#     with torch.no_grad():
#         tx_weight = model.tx_layer.weight.cpu().numpy().T # (130, 4)
#         w_weight = model.spatial_layer.weight.cpu().numpy() # (4, 1)
        
#     # 1. Construct TX prediction using optimized Neural Weights
#     final_tx_pred = np.zeros_like(rx)
#     for ch in range(4):
#         for term_idx, term in enumerate(model_terms):
#             c_idx = term_idx * len(MODEL_LAGS)
#             for lag_idx, lag in enumerate(MODEL_LAGS):
#                 final_tx_pred[:, ch] += tx_weight[c_idx + lag_idx, ch] * shift_signal(term, lag)
                
#     w_n = w_weight / np.linalg.norm(w_weight)
    
#     # 2. Dynamically extract spatial interference from the full signal!
#     res_full = rx - final_tx_pred
#     res_band_full = np.column_stack([score_filter(res_full[:, c]) for c in range(4)])
    
#     e_time_full = res_band_full @ np.conj(w_n) # Shape (N, 1)
#     e_pred_full = e_time_full @ w_n.T          # Shape (N, 4)
    
#     # 3. Final Correction
#     rx_hat = rx - final_tx_pred - e_pred_full
    
#     return rx_hat