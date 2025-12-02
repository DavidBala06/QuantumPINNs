import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # Optional, makes plots prettier

# Configuration
sns.set_theme(style="whitegrid")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRUE_K = 1.0  # The hidden truth
LEARNING_RATE = 0.002
EPOCHS = 3000

# ==========================================
# 1. DATA GENERATION (The Ground Truth)
# ==========================================
def get_synthetic_data(n_samples=2500):
    """
    Generates synthetic training data for a Quantum Harmonic Oscillator.
    True System: V(x) = 0.5 * TRUE_K * x^2
    """
    x = np.random.uniform(-3, 3, n_samples)
    t = np.random.uniform(0, 1, n_samples)
    
    # Analytical Ground State Solution for k=1.0
    # psi(x,t) = exp(-x^2/2) * exp(-i * 0.5 * t)
    u_real = np.exp(-0.5 * x**2) * np.cos(0.5 * t)
    v_imag = -np.exp(-0.5 * x**2) * np.sin(0.5 * t)
    
    return (
        torch.tensor(x, dtype=torch.float32).view(-1, 1).to(DEVICE),
        torch.tensor(t, dtype=torch.float32).view(-1, 1).to(DEVICE),
        torch.tensor(u_real, dtype=torch.float32).view(-1, 1).to(DEVICE),
        torch.tensor(v_imag, dtype=torch.float32).view(-1, 1).to(DEVICE)
    )

# ==========================================
# 2. MODEL DEFINITION
# ==========================================
class QuantumPINN(nn.Module):
    def __init__(self):
        super().__init__()
        # The Neural Network (Approximator)
        self.net = nn.Sequential(
            nn.Linear(2, 50), nn.Tanh(),
            nn.Linear(50, 50), nn.Tanh(),
            nn.Linear(50, 50), nn.Tanh(),
            nn.Linear(50, 50), nn.Tanh(),
            nn.Linear(50, 2) # Outputs: Real (u), Imaginary (v)
        )
        
        # The Physics Parameter (The "Discovery")
        # Initial guess is 0.1 (far from TRUE_K = 1.0)
        self.k_param = nn.Parameter(torch.tensor([0.1], dtype=torch.float32))

    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        output = self.net(inputs)
        return output[:, 0:1], output[:, 1:2]

# ==========================================
# 3. PHYSICS LOSS ENGINE
# ==========================================
def calculate_physics_loss(model, x, t):
    """Calculates the residual of the Schrödinger Equation"""
    x.requires_grad = True
    t.requires_grad = True
    
    u, v = model(x, t)
    
    # First Derivatives
    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    v_t = torch.autograd.grad(v, t, torch.ones_like(v), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    v_x = torch.autograd.grad(v, x, torch.ones_like(v), create_graph=True)[0]
    
    # Second Derivatives
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    v_xx = torch.autograd.grad(v_x, x, torch.ones_like(v_x), create_graph=True)[0]
    
    # Potential V(x) based on the AI's current guess of k
    V_potential = 0.5 * model.k_param * (x**2)
    
    # Schrödinger Residuals
    f_u = u_t + 0.5 * v_xx - V_potential * v
    f_v = v_t - 0.5 * u_xx + V_potential * u
    
    return torch.mean(f_u**2 + f_v**2)

# ==========================================
# 4. TRAINING & VISUALIZATION
# ==========================================
def train():
    print(f"🚀 Starting Training on {DEVICE}...")
    model = QuantumPINN().to(DEVICE)
    optimizer = torch.optim.Adam(list(model.net.parameters()) + [model.k_param], lr=LEARNING_RATE)
    
    x_obs, t_obs, u_obs, v_obs = get_synthetic_data()
    
    history = {'loss': [], 'k': []}
    
    # --- Training Loop ---
    for epoch in range(EPOCHS + 1):
        optimizer.zero_grad()
        
        # Data Loss
        u_pred, v_pred = model(x_obs, t_obs)
        loss_data = torch.mean((u_pred - u_obs)**2 + (v_pred - v_obs)**2)
        
        # Physics Loss (Collocation points)
        x_phy = torch.rand_like(x_obs) * 6 - 3
        t_phy = torch.rand_like(t_obs)
        loss_physics = calculate_physics_loss(model, x_phy, t_phy)
        
        total_loss = loss_data + loss_physics
        total_loss.backward()
        optimizer.step()
        
        # Logging
        history['loss'].append(total_loss.item())
        history['k'].append(model.k_param.item())
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch:04d} | Loss: {total_loss.item():.6f} | Discovered k: {model.k_param.item():.4f}")

    return model, history

def plot_results(model, history):
    """Generates a professional dashboard for GitHub"""
    fig = plt.figure(figsize=(18, 6))
    gs = fig.add_gridspec(1, 3)

    # Plot 1: Parameter Convergence (The Main Result)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(history['k'], linewidth=3, color='#007acc', label='AI Prediction')
    ax1.axhline(y=TRUE_K, color='#d62728', linestyle='--', linewidth=2, label=f'Ground Truth (k={TRUE_K})')
    ax1.set_title("1. Material Discovery (Inverse Problem)", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Parameter $k$ (Potential Stiffness)")
    ax1.legend()
    
    # Plot 2: Loss Function
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(history['loss'], color='#2ca02c', linewidth=2)
    ax2.set_yscale('log')
    ax2.set_title("2. Training Stability (Log Loss)", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Total Loss")

    # Plot 3: Wave Function Check (Verification)
    # Compare True vs Predicted at t=0.5
    ax3 = fig.add_subplot(gs[0, 2])
    test_x = torch.linspace(-3, 3, 100).view(-1, 1).to(DEVICE)
    test_t = torch.ones_like(test_x) * 0.5
    
    with torch.no_grad():
        u_p, v_p = model(test_x, test_t)
        prob_pred = u_p**2 + v_p**2
        
    # Analytical Truth
    x_np = test_x.cpu().numpy()
    u_true = np.exp(-0.5 * x_np**2) * np.cos(0.5 * 0.5)
    v_true = -np.exp(-0.5 * x_np**2) * np.sin(0.5 * 0.5)
    prob_true = u_true**2 + v_true**2
    
    ax3.plot(x_np, prob_true, 'k--', linewidth=2, label='Ground Truth')
    ax3.plot(x_np, prob_pred.cpu().numpy(), 'r-', linewidth=2, alpha=0.8, label='PINN Prediction')
    ax3.set_title("3. Wave Function Fit (t=0.5)", fontsize=14, fontweight='bold')
    ax3.set_xlabel("Position (x)")
    ax3.set_ylabel("Probability Density $|\psi|^2$")
    ax3.legend()

    plt.tight_layout()
    # plt.savefig("pinn_results_dashboard.png", dpi=300)
    # print("✅ Results saved to 'pinn_results_dashboard.png'")
    plt.show()

if __name__ == "__main__":
    trained_model, training_history = train()
    plot_results(trained_model, training_history)