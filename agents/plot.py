import numpy as np 

def moving_average(values, window=50):
    values = np.array(values, dtype=float)
    if len(values) < window:
        window = max(1, len(values))
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")

def save_learning_curve(rewards, path, window=50):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    smooth = moving_average(rewards, window)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(rewards, alpha=0.3, label="reward per episode")
    ax.plot(range(len(rewards) - len(smooth), len(rewards)), smooth, label=f"moving average ({window})")
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Total reward")
    ax.set_title("Coffee Bot learning curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
