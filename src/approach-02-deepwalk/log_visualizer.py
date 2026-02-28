import pandas as pd
import matplotlib.pyplot as plt
import re

def parse_deepwalk_logs(log_file):
    train_data = []
    val_data = []
    
    # Regex patterns for your specific log format
    train_pattern = re.compile(r"Step (\d+) \| Loss: ([\d.]+) \| Pairs: (\d+)")
    val_pattern = re.compile(r"\[VAL\] Step (\d+) \| AUC: ([\d.]+) \| AP: ([\d.]+) \| F1: ([\d.]+)")

    with open(log_file, 'r') as f:
        for line in f:
            t_match = train_pattern.search(line)
            if t_match:
                train_data.append({
                    'Step': int(t_match.group(1)),
                    'Loss': float(t_match.group(2))
                })
            
            v_match = val_pattern.search(line)
            if v_match:
                val_data.append({
                    'Step': int(v_match.group(1)),
                    'AUC': float(v_match.group(2)),
                    'AP': float(v_match.group(3)),
                    'F1': float(v_match.group(4))
                })

    return pd.DataFrame(train_data), pd.DataFrame(val_data)

def plot_metrics(df_train, df_val):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    # Plot 1: Training Loss
    ax1.plot(df_train['Step'], df_train['Loss'], color='tab:red', alpha=0.6, label='Batch Loss')
    # Add a rolling average to see the trend through the noise
    ax1.plot(df_train['Step'], df_train['Loss'].rolling(window=5).mean(), color='darkred', linewidth=2, label='Trend')
    ax1.set_ylabel('Loss')
    ax1.set_title('DeepWalk Training Progress')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()

    # Plot 2: Validation Metrics
    ax2.plot(df_val['Step'], df_val['AUC'], marker='o', label='AUC')
    ax2.plot(df_val['Step'], df_val['AP'], marker='s', label='Avg Precision')
    ax2.plot(df_val['Step'], df_val['F1'], marker='^', label='F1 Score')
    ax2.set_xlabel('Global Step')
    ax2.set_ylabel('Score')
    ax2.set_ylim(0, 1.0)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Ensure the filename matches your uploaded log
    log_path = "logs/deepwalk_training.log"
    df_t, df_v = parse_deepwalk_logs(log_path)
    
    if not df_t.empty:
        plot_metrics(df_t, df_v)
    else:
        print("No data found. Check the log file path and format.")