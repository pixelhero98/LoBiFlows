import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. Configuration & Toy Data Generation
# ==========================================
class LOBConfig:
    input_dim = 40      # 10 levels of Bid/Ask Prices + 10 levels of Volumes = 40 features
    history_len = 50    # Lookback window (T=50)
    hidden_dim = 64     # Size of the context vector
    batch_size = 32
    steps = 1000        # Training steps for demo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_toy_lob_data(batch_size, seq_len, input_dim):
    """
    Generates synthetic "Market History" (context) and "Target" (future state).
    In reality, you would load LOBSTER data here.
    """
    # Random walk to simulate prices + noise for volumes
    history = torch.randn(batch_size, seq_len, input_dim).to(LOBConfig.device).cumsum(dim=1)
    
    # The "Target" is the next step in the sequence (highly correlated with last step of history)
    # We add some non-linear drift to make it interesting for the Neural Net
    last_step = history[:, -1, :]
    target = last_step + 0.1 * torch.sin(last_step) + 0.05 * torch.randn_like(last_step)
    
    return history, target

# ==========================================
# 2. The Model Components
# ==========================================

class ContextEncoder(nn.Module):
    """
    Encodes the time-series history of the Limit Order Book into a fixed vector.
    """
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        
    def forward(self, history):
        # history: [Batch, Seq_Len, Input_Dim]
        # We only care about the final hidden state
        _, (h_n, _) = self.lstm(history)
        return h_n.squeeze(0) # [Batch, Hidden_Dim]

class VectorFieldNetwork(nn.Module):
    """
    Predicts the Velocity Vector 'v' given:
    1. Current position x_t (Noisy LOB)
    2. Time t (Scalar)
    3. Context c (Market History)
    """
    def __init__(self, data_dim, context_dim):
        super().__init__()
        
        # We concatenate [x_t, t, context]
        self.net = nn.Sequential(
            nn.Linear(data_dim + 1 + context_dim, 128),
            nn.SiLU(), # Swish activation is standard for Flows
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, data_dim) # Output is velocity vector of same shape as data
        )

    def forward(self, x, t, context):
        # x: [Batch, Dim]
        # t: [Batch, 1]
        # context: [Batch, Hidden]
        
        # Concatenate inputs
        net_input = torch.cat([x, t, context], dim=1)
        return self.net(net_input)

# ==========================================
# 3. The Conditional MeanFlow Model
# ==========================================

class ConditionalMeanFlow(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = ContextEncoder(config.input_dim, config.hidden_dim)
        self.velocity_net = VectorFieldNetwork(config.input_dim, config.hidden_dim)
        
    def compute_loss(self, x_1, history):
        """
        Computes the Flow Matching Loss.
        x_1: The real future data (Target)
        history: The past market data (Conditioning)
        """
        batch_size = x_1.shape[0]
        
        # 1. Get Context
        context = self.encoder(history)
        
        # 2. Sample random time t ~ Uniform[0, 1]
        t = torch.rand(batch_size, 1).to(x_1.device)
        
        # 3. Sample Noise x_0 ~ Normal(0, 1)
        x_0 = torch.randn_like(x_1)
        
        # 4. Interpolate to find x_t (The input to the network)
        # We use the Optimal Transport path (straight line): x_t = (1-t)x_0 + t*x_1
        x_t = (1 - t) * x_0 + t * x_1
        
        # 5. Predict Velocity
        v_pred = self.velocity_net(x_t, t, context)
        
        # 6. Calculate Target Velocity
        # For a straight path, the velocity is simply (Target - Start)
        v_target = x_1 - x_0
        
        # 7. MSE Loss
        loss = torch.mean((v_pred - v_target) ** 2)
        return loss

    @torch.no_grad()
    def generate(self, history, steps=1):
        """
        Generates the next LOB state from history.
        steps=1 : The 'MeanFlow' promise (super fast)
        steps>1 : Standard ODE integration (higher precision)
        """
        batch_size = history.shape[0]
        context = self.encoder(history)
        
        # Start from pure noise
        x_t = torch.randn(batch_size, LOBConfig.input_dim).to(history.device)
        
        # Euler Integration (Solving the ODE)
        dt = 1.0 / steps
        current_t = 0.0
        
        for i in range(steps):
            # Create time tensor
            t_tensor = torch.full((batch_size, 1), current_t).to(history.device)
            
            # Predict velocity
            v = self.velocity_net(x_t, t_tensor, context)
            
            # Update position: x_{t+1} = x_t + v * dt
            x_t = x_t + v * dt
            current_t += dt
            
        return x_t

# ==========================================
# 4. Training Loop
# ==========================================

model = ConditionalMeanFlow(LOBConfig).to(LOBConfig.device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

print("Starting Training...")
loss_history = []

for step in range(LOBConfig.steps):
    # 1. Get Data
    history, target_lob = generate_toy_lob_data(LOBConfig.batch_size, LOBConfig.history_len, LOBConfig.input_dim)
    
    # 2. Optimization
    optimizer.zero_grad()
    loss = model.compute_loss(target_lob, history)
    loss.backward()
    optimizer.step()
    
    loss_history.append(loss.item())
    
    if step % 200 == 0:
        print(f"Step {step}: Loss = {loss.item():.6f}")

print("Training Complete.")

# ==========================================
# 5. Evaluation / Visualization
# ==========================================
# Generate a sample
test_history, test_target = generate_toy_lob_data(5, LOBConfig.history_len, LOBConfig.input_dim)

# 1-Step Generation (Fast MeanFlow)
generated_1step = model.generate(test_history, steps=1)

# 10-Step Generation (Higher Precision)
generated_10step = model.generate(test_history, steps=10)

print("\n--- Evaluation ---")
print("Real LOB (First 3 dims):", test_target[0, :3].cpu().numpy())
print("Gen 1-Step (First 3 dims):", generated_1step[0, :3].cpu().numpy())
print("Gen 10-Step (First 3 dims):", generated_10step[0, :3].cpu().numpy())

# Plot Loss
plt.plot(loss_history)
plt.title("MeanFlow Training Loss")
plt.xlabel("Step")
plt.ylabel("MSE Loss")
plt.show()
