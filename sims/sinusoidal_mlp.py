import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class MLP(nn.Module):
    """Multi-Layer Perceptron for learning sinusoidal function"""
    def __init__(self, input_size=1, hidden_sizes=[64, 32, 16], output_size=1):
        super(MLP, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Create hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

def generate_data(n_samples=1000, A=1.0, f=1.0, phase=0.0, bias=0.0, sigma=0.1, x_range=(-2, 2)):
    """
    Generate training data using the formula:
    y = A * sin(2*pi*f*x + phase) + bias + N(0, sigma^2)
    
    Args:
        n_samples: Number of data points to generate
        A: Amplitude
        f: Frequency
        phase: Phase shift
        bias: Vertical offset
        sigma: Standard deviation of noise
        x_range: Tuple of (min_x, max_x) for x values
    
    Returns:
        x: Input data (n_samples, 1)
        y: Target data (n_samples, 1)
        x_clean: Clean x values for plotting
        y_clean: Clean y values without noise for plotting
    """
    # Generate x values
    x = np.linspace(x_range[0], x_range[1], n_samples).reshape(-1, 1)
    
    # Generate clean y values (without noise)
    y_clean = A * np.sin(2 * np.pi * f * x + phase) + bias
    
    # Add noise
    noise = np.random.normal(0, sigma, x.shape)
    y = y_clean + noise
    
    # Convert to PyTorch tensors
    x_tensor = torch.FloatTensor(x)
    y_tensor = torch.FloatTensor(y)
    
    return x_tensor, y_tensor, x, y_clean

def train_model(model, x_train, y_train, epochs=1000, learning_rate=0.001, batch_size=32):
    """Train the MLP model"""
    # Create data loader
    dataset = TensorDataset(x_train, y_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    model.train()
    losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_x, batch_y in dataloader:
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        
        # Print progress every 100 epochs
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}')
    
    return losses

def plot_results(x_data, y_data, x_clean, y_clean, model, x_range=(-2, 2), losses=None):
    """Plot the results: real function, data points, and learned function"""
    # Generate points for smooth plotting
    x_plot = np.linspace(x_range[0], x_range[1], 1000).reshape(-1, 1)
    x_plot_tensor = torch.FloatTensor(x_plot)
    
    # Get model predictions
    model.eval()
    with torch.no_grad():
        y_pred = model(x_plot_tensor).numpy()
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot data points
    plt.scatter(x_data.numpy(), y_data.numpy(), alpha=0.6, s=20, color='blue', label='Training data (with noise)')
    
    # Plot real function (without noise) as dashed line
    plt.plot(x_clean, y_clean, 'r--', linewidth=2, label='Real function (without noise)')
    
    # Plot learned function
    plt.plot(x_plot, y_pred, 'g-', linewidth=2, label='Learned function')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('MLP Learning Sinusoidal Function')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training Loss')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()

    plt.show()
    

def main():
    """Main function to run the experiment"""
    # Parameters for data generation
    A = 2.0      # Amplitude
    f = 1.5      # Frequency
    phase = 0.5  # Phase shift
    bias = 0.5   # Vertical offset
    sigma = 0.2  # Noise standard deviation
    n_samples = 1000
    x_range = (-0, 1)
    epochs = 1000
    learning_rate = 0.001
    
    print("Generating training data...")
    x_train, y_train, x_clean, y_clean = generate_data(
        n_samples=n_samples, A=A, f=f, phase=phase, 
        bias=bias, sigma=sigma, x_range=x_range
    )
    
    print(f"Generated {n_samples} training samples")
    print(f"Function: y = {A} * sin(2π * {f} * x + {phase}) + {bias} + N(0, {sigma}²)")
    
    # Create model
    model = MLP(input_size=1, hidden_sizes=[64, 32, 16], output_size=1)
    #model = MLP(input_size=1, hidden_sizes=[5, 5, 5], output_size=1)
    print(f"\nModel architecture:")
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Train model
    print("\nTraining model...")
    losses = train_model(model, x_train, y_train, epochs=epochs, learning_rate=learning_rate)
    
    # Plot results
    print("\nPlotting results...")
    plot_results(x_train, y_train, x_clean, y_clean, model, x_range, losses)
    
    print(f"\nFinal training loss: {losses[-1]:.6f}")

if __name__ == "__main__":
    main()
