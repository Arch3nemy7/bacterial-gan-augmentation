import matplotlib.pyplot as plt
import os

def read_metric_file(filepath):
    steps = []
    values = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    steps.append(int(parts[2]))
                    values.append(float(parts[1]))
    except FileNotFoundError:
        print(f"Warning: File not found: {filepath}")
        return [], []
    return steps, values

def main():
    base_dir = "/home/arch3nemy7/Documents/bacterial-gan-augmentation"
    run_id = "7bb341c8b317480f8eeb09d7b880f44f"
    experiment_id = "808284914952110938"
    
    metrics_path = os.path.join(base_dir, "mlruns", experiment_id, run_id, "metrics")
    output_dir = os.path.join(base_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)
    
    d_loss_path = os.path.join(metrics_path, "discriminator_loss")
    g_loss_path = os.path.join(metrics_path, "generator_loss")
    gp_path = os.path.join(metrics_path, "gradient_penalty")
    
    d_steps, d_values = read_metric_file(d_loss_path)
    g_steps, g_values = read_metric_file(g_loss_path)
    gp_steps, gp_values = read_metric_file(gp_path)
    
    if not d_steps:
        print("No data found to plot.")
        return

    plt.figure(figsize=(12, 10))
    
    # Plot 1: Generator and Discriminator Loss
    plt.subplot(2, 1, 1)
    plt.plot(g_steps, g_values, label='Generator Loss', color='blue', alpha=0.7)
    plt.plot(d_steps, d_values, label='Discriminator Loss', color='red', alpha=0.7)
    plt.title('Training Losses (Epoch 1-55)')
    plt.xlabel('Step')
    plt.ylabel('Loss Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Gradient Penalty
    plt.subplot(2, 1, 2)
    plt.plot(gp_steps, gp_values, label='Gradient Penalty', color='green', alpha=0.7)
    plt.title('Gradient Penalty (WGAN-GP)')
    plt.xlabel('Step')
    plt.ylabel('Penalty Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, "loss_graph_epoch_55.png")
    plt.savefig(output_file)
    print(f"Graph saved to {output_file}")

if __name__ == "__main__":
    main()
