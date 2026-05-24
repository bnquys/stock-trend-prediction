"""
analyze_hyperparams.py
════════════════════════════════════════════════════════════════════════════
Ablation Study & Sensitivity Analysis for RL Trading Agent

Công cụ này giúp xác định:
  1. Hyperparameter nào ảnh hưởng nhiều nhất đến hiệu năng (Sensitivity).
  2. Giá trị tối ưu cho từng tham số (Ablation).

Quy trình:
  - Giữ nguyên các tham số khác, thay đổi 1 tham số duy nhất.
  - Chạy train ngắn (50 episodes) để tiết kiệm thời gian.
  - Ghi nhận Val Return, Sharpe, Win Rate.
════════════════════════════════════════════════════════════════════════════
"""
import os, json, time, argparse, yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from train import train, load_cfg

# Danh sách tham số cần test và các dải giá trị
SEARCH_SPACE = {
    "lr":           [0.0001, 0.0003, 0.0005, 0.001],
    "gamma":        [0.90, 0.95, 0.97, 0.99],
    "batch_size":   [64, 128, 256, 512],
    "eps_decay":    [0.99, 0.995, 0.997, 0.999],
    "tau":          [0.001, 0.005, 0.01, 0.05],
}

def run_study(base_cfg_path: str, episodes: int = 50):
    base_cfg = load_cfg(base_cfg_path)
    results = []
    
    os.makedirs("logs/ablation", exist_ok=True)
    
    for param, values in SEARCH_SPACE.items():
        print(f"\n{'='*60}")
        print(f" Testing Parameter: {param}")
        print(f"{'='*60}")
        
        for val in values:
            print(f"\n>>> Running with {param} = {val}...")
            
            # Clone config và ghi đè giá trị
            cfg = json.loads(json.dumps(base_cfg)) # Deep copy
            cfg["agent"][param] = val
            
            # Chạy training ngắn
            start_t = time.time()
            try:
                # Mocking train function call to return best metrics
                # Trong thực tế, bạn sẽ chạy train(cfg, n_ep_override=episodes)
                # Ở đây tôi giả định train() sẽ được gọi và lưu log.
                train(cfg, n_ep_override=episodes)
                
                # Load kết quả từ training_log.json
                with open("logs/training_log.json", "r") as f:
                    history = json.load(f)
                
                # Lấy trung bình 10 ep cuối làm kết quả
                last_10 = history[-10:]
                avg_ret = np.mean([h["val_return"] for h in last_10])
                avg_sh  = np.mean([h["val_sharpe"] for h in last_10])
                avg_wr  = np.mean([h["val_winrate"] for h in last_10])
                
                res = {
                    "parameter": param,
                    "value": val,
                    "val_return": float(avg_ret),
                    "val_sharpe": float(avg_sh),
                    "val_winrate": float(avg_wr),
                    "time_sec": time.time() - start_t
                }
                results.append(res)
                print(f" Result: Return={avg_ret:.2f}%, Sharpe={avg_sh:.3f}, WR={avg_wr:.1f}%")
                
            except Exception as e:
                print(f" Error testing {param}={val}: {e}")

    # Save results
    df = pd.DataFrame(results)
    df.to_csv("logs/ablation/ablation_results.csv", index=False)
    
    # Generate Plots
    _plot_results(df)
    print(f"\n[Done] Results saved to logs/ablation/")

def _plot_results(df):
    params = df["parameter"].unique()
    fig, axes = plt.subplots(len(params), 1, figsize=(10, 4 * len(params)))
    if len(params) == 1: axes = [axes]
    
    for i, p in enumerate(params):
        sub = df[df["parameter"] == p]
        axes[i].plot(sub["value"].astype(str), sub["val_return"], marker='o', label='Return %')
        axes[i].set_title(f"Impact of {p}")
        axes[i].set_ylabel("Avg Val Return (%)")
        axes[i].grid(True, alpha=0.3)
        
        ax2 = axes[i].twinx()
        ax2.plot(sub["value"].astype(str), sub["val_sharpe"], marker='s', color='orange', label='Sharpe')
        ax2.set_ylabel("Sharpe Ratio")
        
    plt.tight_layout()
    plt.savefig("logs/ablation/ablation_study.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--episodes", type=int, default=50)
    args = parser.parse_args()
    
    run_study(args.config, args.episodes)
