import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# ==========================================
# 1. Configuration
# ==========================================
class LOBConfig:
    input_dim = 40      # Bid/Ask Levels + Volumes
    history_len = 50    
    hidden_dim = 64     
    batch_size = 64
    steps = 2000        # More steps to converge the cycle loss
    lambda_cycle = 2.0  # Weight for the Cycle Consistency Loss
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. Shared Components
# ==========================================
class ContextEncoder(nn.Module):
    """Encodes market history into a context vector."""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
    def forward(self, history):
        _, (h_n, _) = self.lstm(history)
        return h_n.squeeze(0)

class VelocityNet(nn.Module):
    """
    Standard MLP that predicts velocity v(x, t, c).
    Used for BOTH Forward (Encoder) and Reverse (Generator) flows.
    """
    def __init__(self, data_dim, context_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(data_dim + 1 + context_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, data_dim)
        )
    def forward(self, x, t, context):
        return self.net(torch.cat([x, t, context], dim=1))

# ==========================================
# 3. The BiFlow Model (Encoder + Generator)
# ==========================================
class BiFlowLOB(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.context_encoder = ContextEncoder(config.input_dim, config.hidden_dim)
        
        # Generator: Noise -> Data (v = Data - Noise)
        self.generator_flow = VelocityNet(config.input_dim, config.hidden_dim)
        
        # Encoder: Data -> Noise (v = Noise - Data)
        self.encoder_flow = VelocityNet(config.input_dim, config.hidden_dim)
        
    def flow_step(self, net, x_start, context, dt=1.0):
        """Helper to simulate 1-step integration (MeanFlow style)"""
        batch_size = x_start.shape[0]
        t = torch.zeros(batch_size, 1).to(x_start.device) # Start at t=0
        v = net(x_start, t, context)
        return x_start + v * dt

    def compute_loss(self, x_real, history):
        batch_size = x_real.shape[0]
        context = self.context_encoder(history)
        
        # --- A. Standard Flow Matching Loss (Train Generator) ---
        z_noise = torch.randn_like(x_real)
        t = torch.rand(batch_size, 1).to(x_real.device)
        
        # Interpolate: z -> x
        x_t_gen = (1 - t) * z_noise + t * x_real
        v_pred_gen = self.generator_flow(x_t_gen, t, context)
        v_target_gen = x_real - z_noise
        loss_gen = torch.mean((v_pred_gen - v_target_gen) ** 2)

        # --- B. Standard Flow Matching Loss (Train Encoder) ---
        # Interpolate: x -> z (Time flows backwards conceptually, or just target z)
        # We model the path from Data(0) to Noise(1)
        x_t_enc = (1 - t) * x_real + t * z_noise
        v_pred_enc = self.encoder_flow(x_t_enc, t, context)
        v_target_enc = z_noise - x_real
        loss_enc = torch.mean((v_pred_enc - v_target_enc) ** 2)
        
        # --- C. Cycle Consistency Loss (The BiFlow Magic) ---
        # 1. Encode: Real Data -> Latent Z (1-step estimate)
        z_pred = self.flow_step(self.encoder_flow, x_real, context)
        
        # 2. Decode: Latent Z -> Reconstructed Data
        x_recon = self.flow_step(self.generator_flow, z_pred, context)
        
        # 3. Loss: Did we get back to the start?
        loss_cycle = torch.mean((x_real - x_recon) ** 2)
        
        return loss_gen + loss_enc + (LOBConfig.lambda_cycle * loss_cycle)

    @torch.no_grad()
    def generate(self, history):
        """Generates LOB state from history (Noise -> Data)"""
        context = self.context_encoder(history)
        z = torch.randn(history.shape[0], LOBConfig.input_dim).to(history.device)
        return self.flow_step(self.generator_flow, z, context)
    
    @torch.no_grad()
    def analyze_market_state(self, current_lob, history):
        """
        Inverse: Maps a real LOB state to Latent Noise.
        Useful for anomaly detection: if z is far from Gaussian, it's a crash/shock.
        """
        context = self.context_encoder(history)
        z = self.flow_step(self.encoder_flow, current_lob, context)
        return z

# ==========================================
# 4. Training Loop
# ==========================================
def generate_toy_lob(bs, steps, dim):
    # Simple toy data: Random walk
    hist = torch.randn(bs, steps, dim).cumsum(dim=1).to(LOBConfig.device)
    target = hist[:, -1, :] + torch.randn(bs, dim).to(LOBConfig.device)*0.1
    return hist, target

model = BiFlowLOB(LOBConfig).to(LOBConfig.device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
losses = []

print("Training BiFlow LOB Model...")
for step in range(LOBConfig.steps):
    hist, target = generate_toy_lob(LOBConfig.batch_size, LOBConfig.history_len, LOBConfig.input_dim)
    
    optimizer.zero_grad()
    loss = model.compute_loss(target, hist)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    
    if step % 500 == 0:
        print(f"Step {step}: Loss {loss.item():.5f}")

# ==========================================
# 5. Verification: Cycle Check
# ==========================================
print("\n--- Cycle Consistency Check ---")
test_hist, test_real = generate_toy_lob(5, 50, 40)

# 1. Real -> Z
z_latent = model.analyze_market_state(test_real, test_hist)
# 2. Z -> Reconstruction
x_recon = model.generate(test_hist) # Note: technically we should pass z_latent here to test cycle strictly

# Strict cycle test manually:
ctx = model.context_encoder(test_hist)
z_inferred = model.flow_step(model.encoder_flow, test_real, ctx)
x_recovered = model.flow_step(model.generator_flow, z_inferred, ctx)

print(f"Real Price (Dim 0): {test_real[0,0]:.4f}")
print(f"Reconstructed:      {x_recovered[0,0]:.4f}")
print(f"Reconstruction Error: {torch.mean((test_real - x_recovered)**2):.6f}")

plt.plot(losses)
plt.title("BiFlow Training Loss")
plt.show()
