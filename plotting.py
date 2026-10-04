import json
import numpy as np
import matplotlib.pyplot as plt

def load_diagnostics(model_paths):
    """
    Loads JSON telemetry files into structured dictionaries.
    model_paths: dict of {"Display Name": "path/to/file.json"}
    """
    dataset = {}
    for name, path in model_paths.items():
        with open(path, "r") as f:
            raw = json.load(f)
        
        # Handle cases where records are stored under a top-level key
        records = raw if isinstance(raw, list) else raw.get("records", raw.get("history", []))
        
        grad_norms = []
        for r in records:
            p_grads = r.get("param_gradients", {})
            if isinstance(p_grads, dict):
                # Calculate global L2 norm from individual parameter norms
                sq_sum = sum(v ** 2 for v in p_grads.values() if isinstance(v, (int, float)))
                grad_norms.append(np.sqrt(sq_sum))
            elif isinstance(p_grads, (int, float)):
                grad_norms.append(float(p_grads))
            else:
                grad_norms.append(0.0)

        dataset[name] = {
            "step_loss": np.array([r.get("step_loss", 0.0) for r in records]),
            "window_idx": np.array([r.get("window_idx", 0) for r in records]),
            "slot_losses": np.array([r.get("slot_losses", []) for r in records if "slot_losses" in r]),
            "grad_norm": np.array(grad_norms)
        }
    return dataset

def plot_loss_trajectory(data, ax=None, smooth_window=10):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    
    for name, metrics in data.items():
        losses = metrics["step_loss"]
        steps = np.arange(1, len(losses) + 1)
        
        # Plot raw telemetry with light alpha
        ax.plot(steps, losses, alpha=0.25)
        
        # Plot smoothed rolling mean
        if len(losses) >= smooth_window:
            kernel = np.ones(smooth_window) / smooth_window
            smoothed = np.convolve(losses, kernel, mode="valid")
            ax.plot(np.arange(smooth_window, len(losses) + 1), smoothed, label=name, linewidth=2)
        else:
            ax.plot(steps, losses, label=name, linewidth=2)
            
    ax.set_title("Training Loss Trajectory (Window Steps)")
    ax.set_xlabel("Window Step")
    ax.set_ylabel("Cross Entropy Loss")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

def plot_position_error_profile(data, ax=None, last_n=100):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
        
    for name, metrics in data.items():
        slots = metrics["slot_losses"]
        if len(slots) == 0:
            continue
            
        # Average per-token loss across the final windows
        recent_slots = slots[-last_n:]
        mean_profile = recent_slots.mean(axis=0)
        positions = np.arange(len(mean_profile))
        
        ax.plot(positions, mean_profile, label=name, linewidth=2)
        
    ax.set_title(f"Per-Token Position Loss (Average Over Final {last_n} Steps)")
    ax.set_xlabel("Token Position Inside Window (0 to Window Size)")
    ax.set_ylabel("Mean Cross Entropy")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

def plot_window_progression(data, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
        
    for name, metrics in data.items():
        losses = metrics["step_loss"]
        w_idx = metrics["window_idx"]
        
        unique_indices = np.unique(w_idx)
        avg_losses = []
        for idx in unique_indices:
            mask = w_idx == idx
            avg_losses.append(losses[mask].mean() if np.any(mask) else 0.0)
            
        ax.plot(unique_indices, avg_losses, marker="o", linewidth=2, label=name)
        
    ax.set_title("Loss by Window Progression (Memory Buildup)")
    ax.set_xlabel("Window Index in Sequence")
    ax.set_ylabel("Mean Loss")
    ax.set_xticks(unique_indices)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

def plot_gradient_flow(data, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
        
    for name, metrics in data.items():
        norms = metrics["grad_norm"]
        if np.all(norms == 0.0):
            continue
        ax.plot(np.arange(1, len(norms) + 1), norms, label=name, alpha=0.85)
        
    ax.set_title("Gradient Norm Stability")
    ax.set_xlabel("Window Step")
    ax.set_ylabel("Total Gradient L2 Norm")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

def plot_layer_gradients(model_path, ax=None, last_n=20):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
        
    with open(model_path, "r") as f:
        records = json.load(f)
    if not isinstance(records, list):
        records = records.get("records", records.get("history", []))
        
    recent = records[-last_n:]
    all_keys = list(recent[0].get("param_gradients", {}).keys())
    
    # Compute mean gradient norm per parameter over recent windows
    avg_norms = {}
    for k in all_keys:
        vals = [r.get("param_gradients", {}).get(k, 0.0) for r in recent]
        avg_norms[k] = np.mean(vals)
        
    # Group into summary categories
    categories = {"F_stream": [], "L_stream": [], "B_stream": [], "attn": [], "ffn": [], "other": []}
    for k, val in avg_norms.items():
        if ".F." in k:
            categories["F_stream"].append(val)
        elif ".L." in k:
            categories["L_stream"].append(val)
        elif ".B." in k:
            categories["B_stream"].append(val)
        elif "attn" in k:
            categories["attn"].append(val)
        elif "ffn" in k:
            categories["ffn"].append(val)
        else:
            categories["other"].append(val)
            
    group_means = {k: np.mean(v) if len(v) > 0 else 0.0 for k, v in categories.items()}
    ax.bar(group_means.keys(), group_means.values(), color="steelblue")
    ax.set_title(f"Average Gradient Norm by Stream/Component (Final {last_n} Windows)")
    ax.set_ylabel("Mean L2 Norm")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)

def plot_diagnostic_dashboard(model_paths, save_path=None):
    data = load_diagnostics(model_paths)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    plot_loss_trajectory(data, ax=axes[0, 0])
    plot_position_error_profile(data, ax=axes[0, 1])
    plot_window_progression(data, ax=axes[1, 0])
    plot_gradient_flow(data, ax=axes[1, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Comparison dashboard saved to {save_path}")
        
    plt.show()