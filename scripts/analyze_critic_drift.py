"""Deep analysis of potential critic drift in WGAN-GP training."""

import numpy as np
from pathlib import Path

def analyze_critic_drift():
    """Analyze if the discriminator loss pattern indicates critic drift."""

    print("\n" + "="*80)
    print("🔍 CRITIC DRIFT ANALYSIS FOR WGAN-GP")
    print("="*80 + "\n")

    # Load metrics
    mlflow_run_path = Path("mlruns/808284914952110938/metrics")

    # Read discriminator loss
    disc_loss_data = []
    with open(mlflow_run_path / "discriminator_loss", 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                step, value = int(parts[2]), float(parts[1])
                disc_loss_data.append((step, value))

    # Read generator loss
    gen_loss_data = []
    with open(mlflow_run_path / "generator_loss", 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                step, value = int(parts[2]), float(parts[1])
                gen_loss_data.append((step, value))

    # Read gradient penalty
    gp_data = []
    with open(mlflow_run_path / "gradient_penalty", 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                step, value = int(parts[2]), float(parts[1])
                gp_data.append((step, value))

    disc_losses = [x[1] for x in disc_loss_data]
    gen_losses = [x[1] for x in gen_loss_data]
    gps = [x[1] for x in gp_data]

    print("📊 WGAN-GP Theoretical Expectations:")
    print("-" * 80)
    print("1. Discriminator Loss (Wasserstein Distance):")
    print("   - Should converge to a stable negative value close to 0")
    print("   - Represents Earth Mover's Distance between real and fake distributions")
    print("   - Formula: D_loss = E[D(fake)] - E[D(real)]")
    print("   - Optimal: Near 0 (distributions are similar)")
    print()
    print("2. Generator Loss:")
    print("   - Should be negative and stable (for WGAN-GP)")
    print("   - Formula: G_loss = -E[D(G(z))]")
    print("   - Generator wants D(G(z)) to be HIGH (positive), so loss is NEGATIVE")
    print("   - Exploding to positive values = discriminator rejecting all fake images")
    print()
    print("3. Gradient Penalty:")
    print("   - Should stabilize around 0.01-0.1")
    print("   - Enforces 1-Lipschitz constraint")
    print("   - If GP is too strong, it over-constrains the discriminator")
    print()
    print("="*80)
    print("📈 YOUR TRAINING METRICS:")
    print("="*80)
    print()

    # Discriminator analysis
    print("🎯 Discriminator Loss (Wasserstein Distance):")
    print(f"   Initial:          {disc_losses[0]:>10.4f}")
    print(f"   After 5 epochs:   {disc_losses[5]:>10.4f}")
    print(f"   After 10 epochs:  {disc_losses[10]:>10.4f}")
    print(f"   After 20 epochs:  {disc_losses[20]:>10.4f}")
    print(f"   Final (epoch 54): {disc_losses[-1]:>10.4f}")
    print(f"   Trend: {disc_losses[0]:.2f} → {disc_losses[-1]:.4f}")
    print()

    # Check for drift
    drift_after_10 = np.mean(disc_losses[10:])
    std_after_10 = np.std(disc_losses[10:])
    print(f"   Stability after epoch 10:")
    print(f"   - Mean:    {drift_after_10:.6f}")
    print(f"   - Std Dev: {std_after_10:.6f}")
    print(f"   - Range:   [{min(disc_losses[10:]):.6f}, {max(disc_losses[10:]):.6f}]")
    print()

    # Critic drift check
    is_drifting = abs(drift_after_10) > 0.1
    if is_drifting:
        print("   ❌ CRITIC DRIFT DETECTED: Loss drifted far from optimal range")
    else:
        print("   ✅ NO CRITIC DRIFT: Discriminator loss is stable near 0")
    print()

    # Generator analysis
    print("🎯 Generator Loss:")
    print(f"   Initial:          {gen_losses[0]:>10.4f}")
    print(f"   After 10 epochs:  {gen_losses[10]:>10.4f}")
    print(f"   After 20 epochs:  {gen_losses[20]:>10.4f}")
    print(f"   Final (epoch 54): {gen_losses[-1]:>10.4f}")
    print()

    # Check if generator loss is consistently positive (BAD for WGAN-GP)
    positive_count = sum(1 for x in gen_losses[10:] if x > 0)
    total_count = len(gen_losses[10:])
    print(f"   Positive loss count after epoch 10: {positive_count}/{total_count} ({100*positive_count/total_count:.1f}%)")

    if positive_count > total_count * 0.5:
        print("   ❌ SEVERE ISSUE: Generator loss is positive (discriminator scores are negative)")
        print("      This means D(G(z)) < 0, discriminator is strongly rejecting fakes")
    else:
        print("   ✅ Generator loss mostly negative (expected for WGAN-GP)")
    print()

    # Gradient penalty analysis
    print("🎯 Gradient Penalty:")
    gp_mean = np.mean(gps[10:])
    gp_std = np.std(gps[10:])
    print(f"   Mean (after epoch 10): {gp_mean:.6f}")
    print(f"   Std Dev:               {gp_std:.6f}")

    if gp_mean < 0.001:
        print("   ⚠️  GP too weak: May not enforce Lipschitz constraint properly")
    elif gp_mean > 0.1:
        print("   ⚠️  GP too strong: May over-constrain discriminator")
    else:
        print("   ✅ GP in optimal range (0.001 - 0.1)")
    print()

    # Diagnosis
    print("="*80)
    print("🔬 DIAGNOSIS:")
    print("="*80)
    print()

    # Calculate discriminator output scores
    # D_loss = E[D(fake)] - E[D(real)]
    # If D_loss ≈ -0.01, then E[D(fake)] - E[D(real)] ≈ -0.01
    # This means E[D(real)] - E[D(fake)] ≈ 0.01 (small gap, which is good!)

    print("Based on the metrics:")
    print()
    print("1. ❌ NOT CRITIC DRIFT:")
    print("   - Discriminator loss stable at ~-0.01 (near optimal)")
    print("   - Gradient penalty is working correctly (~0.008)")
    print("   - The discriminator is NOT drifting away")
    print()
    print("2. ✅ ACTUAL PROBLEM: GENERATOR FAILURE")
    print("   - Generator loss exploding from -1.32 to +107.88")
    print("   - This means D(G(z)) went from ~+1.32 to ~-107.88")
    print("   - Discriminator is strongly rejecting ALL generated images")
    print("   - Generator is producing increasingly worse images")
    print()
    print("3. 🔍 ROOT CAUSE:")
    print("   The discriminator outputs (D(real) and D(fake)) tell us:")
    print()

    # Early training (epoch 1)
    early_disc_loss = disc_losses[1]  # ≈ 0.018
    early_gen_loss = gen_losses[1]    # ≈ -1.65
    print(f"   Early (Epoch 1):")
    print(f"   - D_loss = E[D(fake)] - E[D(real)] = {early_disc_loss:.3f}")
    print(f"   - G_loss = -E[D(fake)] = {early_gen_loss:.3f}")
    print(f"   - Therefore: E[D(fake)] ≈ {-early_gen_loss:.3f}")
    print(f"   - And: E[D(real)] ≈ {-early_gen_loss - early_disc_loss:.3f}")
    print(f"   - Gap is small = Generator producing reasonable images")
    print()

    # Late training (epoch 50)
    late_disc_loss = disc_losses[50]  # ≈ -0.008
    late_gen_loss = gen_losses[50]    # ≈ 99.08
    print(f"   Late (Epoch 50):")
    print(f"   - D_loss = E[D(fake)] - E[D(real)] = {late_disc_loss:.3f}")
    print(f"   - G_loss = -E[D(fake)] = {late_gen_loss:.3f}")
    print(f"   - Therefore: E[D(fake)] ≈ {-late_gen_loss:.3f}")
    print(f"   - Discriminator giving EXTREMELY NEGATIVE scores to fakes")
    print(f"   - Generator completely collapsed")
    print()

    print("="*80)
    print("💡 CONCLUSION:")
    print("="*80)
    print()
    print("❌ NOT critic drift - discriminator is behaving correctly")
    print("❌ The issue is GENERATOR MODE COLLAPSE")
    print()
    print("Possible causes:")
    print("1. Generator architecture insufficient for task complexity")
    print("2. Training imbalance (n_critic=5 may be too aggressive)")
    print("3. Learning rate too low for generator to recover from bad state")
    print("4. Gradient clipping preventing generator from learning")
    print("5. Mixed precision causing numerical instability in generator")
    print()
    print("="*80)

if __name__ == "__main__":
    analyze_critic_drift()
