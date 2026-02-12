import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from demo_model import BiFlowLOB

class LOBConfig:
    input_dim = 2       # [Return, Imbalance]
    history_len = 50    
    hidden_dim = 64     
    batch_size = 64
    steps = 4000        
    lambda_cycle = 2.0  
    return_scale = 100.0 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_causal_lob(batch_size, seq_len):
    """
    Generates Ground Truth where Imbalance CAUSES Returns.
    """
    # 1. Imbalance (-1 to 1)
    imbalance = torch.rand(batch_size, seq_len, 1).to(LOBConfig.device) * 2 - 1
    
    # 2. Returns (0.2 * Lagged_Imbalance + Noise)
    noise = torch.randn(batch_size, seq_len, 1).to(LOBConfig.device) * 0.05
    shifted_imbalance = torch.roll(imbalance, shifts=1, dims=1)
    shifted_imbalance[:, 0, :] = 0
    returns = 0.2 * shifted_imbalance + noise
    
    # 3. Concatenate [Return, Imbalance]
    # Scale returns for numerical stability
    returns_scaled = returns * LOBConfig.return_scale
    return torch.cat([returns_scaled, imbalance], dim=2)

def returns_to_prices(returns_scaled, start_price=100.0):
    """Helper to convert scaled returns back to price path"""
    real_returns = returns_scaled / LOBConfig.return_scale
    log_path = torch.cumsum(real_returns, dim=0)
    return start_price * torch.exp(log_path)

# --- A. TRAIN ---
print("1. Training...")
model = BiFlowLOB(LOBConfig).to(LOBConfig.device)
opt = optim.Adam(model.parameters(), lr=0.001)

for i in range(LOBConfig.steps):
    data = generate_causal_lob(LOBConfig.batch_size, LOBConfig.history_len + 1)
    opt.zero_grad()
    loss = model.compute_loss(data[:, -1], data[:, :-1])
    loss.backward()
    opt.step()

# --- B. GENERATE EVALUATION DATA ---
print("2. Generating Scenarios...")

# 1. Trajectory & Distribution Data
# Generate a long path autoregressively
seq_len = 100
hist = generate_causal_lob(1, LOBConfig.history_len)
gen_path = []
curr_hist = hist.clone()

for _ in range(seq_len):
    next_step = model.generate_step(curr_hist)
    gen_path.append(next_step)
    curr_hist = torch.cat([curr_hist[:, 1:], next_step.unsqueeze(1)], dim=1)

gen_tensor = torch.stack(gen_path).squeeze(1).cpu()
real_tensor = generate_causal_lob(1, seq_len).squeeze(0).cpu() # Independent real sample

# 2. Causal Intervention Data
# Twin A (Buy) vs Twin B (Sell)
base_hist = generate_causal_lob(1000, LOBConfig.history_len)
hist_buy = base_hist.clone(); hist_buy[:, -1, 1] = 1.0  # Force Buy Imbalance
hist_sell = base_hist.clone(); hist_sell[:, -1, 1] = -1.0 # Force Sell Imbalance

future_buy = model.generate_step(hist_buy).cpu().numpy()[:, 0]
future_sell = model.generate_step(hist_sell).cpu().numpy()[:, 0]

# --- C. VISUALIZE DASHBOARD ---
print("3. Plotting Dashboard...")
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
plt.subplots_adjust(hspace=0.3)

# Plot 1: Trajectory (Price Path)
ax = axes[0, 0]
real_price = returns_to_prices(real_tensor[:, 0])
gen_price = returns_to_prices(gen_tensor[:, 0])
ax.plot(real_price, label='Real Path (Random)', color='black', alpha=0.6)
ax.plot(gen_price, label='Generated Path', color='dodgerblue', linestyle='--')
ax.set_title("1. Autoregressive Price Path")
ax.legend()
ax.grid(alpha=0.3)

# Plot 2: Volatility Distribution
ax = axes[0, 1]
sns.kdeplot(real_tensor[:, 0], ax=ax, fill=True, label='Real Volatility', color='gray')
sns.kdeplot(gen_tensor[:, 0], ax=ax, fill=True, label='Generated Volatility', color='dodgerblue')
ax.set_title("2. Returns Distribution (Stylized Facts)")
ax.set_xlim(-30, 30) # Zoom in on the bell curve
ax.legend()
ax.grid(alpha=0.3)

# Plot 3: Causal Intervention
ax = axes[1, 0]
sns.kdeplot(future_buy, ax=ax, fill=True, label='Condition: Heavy BUY', color='green')
sns.kdeplot(future_sell, ax=ax, fill=True, label='Condition: Heavy SELL', color='red')
ax.axvline(0, color='black', linestyle='--')
ax.set_title("3. Causal Check (Does Order Flow Move Price?)")
ax.legend()
ax.grid(alpha=0.3)

# Plot 4: Correlation Check (Scatter)
ax = axes[1, 1]
# Check if generated returns correlate with the imbalance we fed it
# We take the generated sequence: X=Imbalance, Y=Return
# Note: The model generates both, so we check internal consistency
gen_imb = gen_tensor[:, 1].numpy()
gen_ret = gen_tensor[:, 0].numpy()
ax.scatter(gen_imb, gen_ret, alpha=0.5, color='purple', s=10)
ax.set_title("4. Learned Correlation (Imbalance vs Return)")
ax.set_xlabel("Generated Imbalance")
ax.set_ylabel("Generated Return")
ax.grid(alpha=0.3)

plt.show()
