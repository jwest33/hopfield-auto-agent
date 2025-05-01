#!/usr/bin/env python3
"""agent_state_viewer.py – inspect a saved *agent_state.npz*
===========================================================
An enhanced CLI tool for analyzing agent state files produced by the 
experience-based learning Hopfield agent.

Features:
- View basic agent state (position, energy, hunger, etc.)
- Analyze learning data (cell experiences, Q-values)
- Visualize memory patterns with optional plots
- Export data to CSV for further analysis

Usage
-----
$ python agent_state_viewer.py                # looks for ./agent_state.npz
$ python agent_state_viewer.py path/to/file   # inspect different file
$ python agent_state_viewer.py --full         # dump all arrays verbatim
$ python agent_state_viewer.py --plot         # visualize memory patterns
$ python agent_state_viewer.py --export       # export data to CSV files
$ python agent_state_viewer.py --learning     # detailed learning analysis
"""
from __future__ import annotations
import argparse
import os
import sys
import datetime as dt
import numpy as np
from textwrap import indent
import pandas as pd
import json
from collections import defaultdict
from typing import Dict, Any, List, Tuple
import matplotlib.pyplot as plt

# Check if tabulate is installed
try:
    from tabulate import tabulate
except ImportError:
    # Fallback implementation if tabulate is not installed
    def tabulate(data, headers, tablefmt=None):
        result = []
        # Add header
        if headers:
            result.append("  ".join(str(h) for h in headers))
            result.append("-" * (sum(len(str(h)) for h in headers) + 2 * (len(headers) - 1)))
        
        # Add rows
        for row in data:
            result.append("  ".join(str(cell) for cell in row))
        
        return "\n".join(result)

# ───────────────────── helpers ─────────────────────

def fmt_bool(b: bool) -> str:
    """Format boolean as yes/no string."""
    return "yes" if b else "no"

def ts_to_str(ts: float) -> str:
    """Convert timestamp to readable datetime string."""
    return dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")

def print_header(title: str):
    """Print a formatted section header."""
    print("\n" + title)
    print("─" * len(title))

def safe_get_dict(data, key_path):
    """Safely extract dictionary from numpy array or direct dict."""
    if key_path not in data:
        return {}
    
    value = data[key_path]
    
    # Try various ways to get a dictionary
    if isinstance(value, dict):
        return value
    elif isinstance(value, np.ndarray):
        if value.size == 0:
            return {}
        
        # Try to extract dictionary from numpy array
        try:
            if isinstance(value[0], dict):
                return value[0]
            elif hasattr(value[0], 'item'):
                extracted = value[0].item()
                if isinstance(extracted, dict):
                    return extracted
        except (IndexError, AttributeError, TypeError):
            pass
    
    # If all else fails, return empty dict
    return {}

def analyze_memory_patterns(memory_matrix: np.ndarray) -> Dict[str, Any]:
    """Analyze memory patterns to extract statistics."""
    if memory_matrix.size == 0:
        return {
            "count": 0,
            "avg_magnitude": 0,
            "max_magnitude": 0,
            "avg_sparsity": 0,
            "memory_density": 0
        }
        
    magnitudes = np.linalg.norm(memory_matrix, axis=1)
    sparsity = np.mean(memory_matrix == 0, axis=1)
    
    return {
        "count": memory_matrix.shape[0],
        "avg_magnitude": float(np.mean(magnitudes)),
        "max_magnitude": float(np.max(magnitudes)) if magnitudes.size > 0 else 0,
        "avg_sparsity": float(np.mean(sparsity)),
        "memory_density": float(memory_matrix.shape[0] / memory_matrix.shape[1] if memory_matrix.shape[1] > 0 else 0)
    }

def plot_memory_patterns(mem0_M: np.ndarray, mem1_M: np.ndarray):
    """Create visualizations of memory patterns."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Hopfield Memory Analysis', fontsize=16)
    
    # Plot memory density for memory layer 0
    if mem0_M.size > 0:
        axes[0, 0].set_title('Memory Layer 0 - Pattern Density')
        im0 = axes[0, 0].imshow(mem0_M[:min(50, mem0_M.shape[0])], aspect='auto', cmap='viridis')
        axes[0, 0].set_xlabel('Feature Dimension')
        axes[0, 0].set_ylabel('Memory Index')
        fig.colorbar(im0, ax=axes[0, 0])
        
        # Plot histogram of values for memory layer 0
        axes[0, 1].set_title('Memory Layer 0 - Value Distribution')
        axes[0, 1].hist(mem0_M.flatten(), bins=50, alpha=0.7)
        axes[0, 1].set_xlabel('Value')
        axes[0, 1].set_ylabel('Frequency')
    
    # Plot memory density for memory layer 1
    if mem1_M.size > 0:
        axes[1, 0].set_title('Memory Layer 1 - Pattern Density')
        im1 = axes[1, 0].imshow(mem1_M[:min(50, mem1_M.shape[0])], aspect='auto', cmap='viridis')
        axes[1, 0].set_xlabel('Feature Dimension')
        axes[1, 0].set_ylabel('Memory Index')
        fig.colorbar(im1, ax=axes[1, 0])
        
        # Plot histogram of values for memory layer 1
        axes[1, 1].set_title('Memory Layer 1 - Value Distribution')
        axes[1, 1].hist(mem1_M.flatten(), bins=50, alpha=0.7)
        axes[1, 1].set_xlabel('Value')
        axes[1, 1].set_ylabel('Frequency')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('memory_analysis.png')
    print(f"Memory visualization saved to 'memory_analysis.png'")

def analyze_learning_data(cell_experience: Dict, q_values: Dict) -> Dict[str, Any]:
    """Analyze learning data to extract insights."""
    results = {
        "cell_type_analysis": {},
        "q_value_analysis": {},
        "action_preferences": defaultdict(int)
    }
    
    # Analyze cell type experiences
    for cell_type, exp in cell_experience.items():
        results["cell_type_analysis"][cell_type] = {
            "reward": exp.get("reward", 0),
            "visits": exp.get("visits", 0),
            "reward_per_visit": exp.get("reward", 0) / exp.get("visits", 1) if exp.get("visits", 0) > 0 else 0
        }
    
    # Analyze Q-values
    max_q_by_state = {}
    if q_values:
        for state, actions in q_values.items():
            max_action = max(actions.items(), key=lambda x: x[1]) if actions else (None, 0)
            max_q_by_state[state] = max_action
            results["action_preferences"][max_action[0]] += 1
    
    # Calculate statistics on Q-values
    q_magnitudes = [v for actions in q_values.values() for v in actions.values()]
    if q_magnitudes:
        results["q_value_analysis"]["count"] = len(q_magnitudes)
        results["q_value_analysis"]["avg"] = np.mean(q_magnitudes)
        results["q_value_analysis"]["min"] = min(q_magnitudes)
        results["q_value_analysis"]["max"] = max(q_magnitudes)
        results["q_value_analysis"]["std"] = np.std(q_magnitudes)
    
    return results

def analyze_history(history: Dict) -> Dict[str, Any]:
    """Analyze agent history to extract trends and statistics."""
    if not history or not all(k in history for k in ["energy", "hunger", "pain"]):
        return {}
    
    results = {}
    
    # Calculate statistics for each metric
    for metric in ["energy", "hunger", "pain", "food_stored"]:
        if metric in history and history[metric]:
            values = history[metric]
            results[metric] = {
                "avg": float(np.mean(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "last": float(values[-1]) if values else 0,
                "trend": "increasing" if len(values) > 10 and np.mean(values[-10:]) > np.mean(values[:10]) else 
                         "decreasing" if len(values) > 10 and np.mean(values[-10:]) < np.mean(values[:10]) else "stable"
            }
    
    # Analyze action distribution
    if "actions" in history and history["actions"]:
        action_counts = defaultdict(int)
        for action in history["actions"]:
            action_counts[action] += 1
        
        results["action_distribution"] = {
            action: {
                "count": count,
                "percentage": count / len(history["actions"]) * 100
            }
            for action, count in action_counts.items()
        }
    
    # Analyze rewards trend
    if "rewards" in history and history["rewards"]:
        rewards = history["rewards"]
        results["rewards"] = {
            "avg": float(np.mean(rewards)),
            "min": float(np.min(rewards)),
            "max": float(np.max(rewards)),
            "last": float(rewards[-1]) if rewards else 0,
            "positive_ratio": sum(1 for r in rewards if r > 0) / len(rewards) if rewards else 0,
            "trend": "improving" if len(rewards) > 10 and np.mean(rewards[-10:]) > np.mean(rewards[:10]) else 
                     "worsening" if len(rewards) > 10 and np.mean(rewards[-10:]) < np.mean(rewards[:10]) else "stable"
        }
    
    return results

def export_to_csv(data, filename_prefix: str = "agent_data"):
    """Export agent data to CSV files for further analysis."""
    # Export cell experience data
    cell_exp = safe_get_dict(data, "cell_experience")
    if cell_exp:
        cell_exp_df = pd.DataFrame([
            {
                "cell_type": cell_type,
                "reward": exp.get("reward", 0),
                "visits": exp.get("visits", 0),
                "last_visit": exp.get("last_visit", 0)
            }
            for cell_type, exp in cell_exp.items()
        ])
        cell_exp_df.to_csv(f"{filename_prefix}_cell_experience.csv", index=False)
        print(f"Cell experience data exported to {filename_prefix}_cell_experience.csv")
    
    # Export history data
    history = safe_get_dict(data, "history")
    if history:
        # Ensure all arrays are the same length
        arrays = [v for k, v in history.items() 
                 if k in ["energy", "hunger", "pain", "food_stored", "rewards"] 
                 and isinstance(v, (list, np.ndarray))]
        
        if arrays:
            min_length = min(len(v) for v in arrays)
            history_df = pd.DataFrame({
                k: v[-min_length:] if isinstance(v, (list, np.ndarray)) and len(v) >= min_length else None
                for k, v in history.items()
                if k in ["energy", "hunger", "pain", "food_stored", "rewards"]
            })
            
            if not history_df.empty:
                history_df.to_csv(f"{filename_prefix}_history.csv", index=False)
                print(f"History data exported to {filename_prefix}_history.csv")
    
    # Export Q-values data (as json because of complex structure)
    q_values = safe_get_dict(data, "q_values")
    if q_values:
        with open(f"{filename_prefix}_q_values.json", "w") as f:
            json.dump(q_values, f, indent=2)
        print(f"Q-values data exported to {filename_prefix}_q_values.json")

def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Enhanced Hopfield Agent State Analyzer")
    parser.add_argument("file", nargs="?", default="agent_state.npz",
                        help="path to the .npz file (default: agent_state.npz)")
    parser.add_argument("--full", action="store_true", help="dump full arrays verbatim")
    parser.add_argument("--plot", action="store_true", help="create visualizations of memory patterns")
    parser.add_argument("--export", action="store_true", help="export data to CSV files for further analysis")
    parser.add_argument("--learning", action="store_true", help="show detailed learning analysis")
    args = parser.parse_args(argv)

    if not os.path.exists(args.file):
        print(f"Error: '{args.file}' not found.", file=sys.stderr)
        sys.exit(1)

    try:
        data = np.load(args.file, allow_pickle=True)
    except Exception as e:
        print(f"Error loading file: {e}", file=sys.stderr)
        sys.exit(1)

    # ───── basic physiology & position ─────
    print_header("Physiology & Position")
    print(f"position : {data['pos'].tolist()}")
    print(f"energy   : {float(data['energy']):.1f}")
    print(f"hunger   : {int(data['hunger'])}")
    print(f"pain     : {int(data['pain'])}")
    print(f"carrying : {fmt_bool(bool(data['carrying']))}")
    print(f"stored   : {int(data['store'])} food items")
    
    # Print tick count if available
    if "tick_count" in data:
        print(f"ticks    : {int(data['tick_count'])}")

    # ───── memory layer summaries ─────
    mem0_M, mem0_t = data['mem0_M'], data['mem0_t']
    print_header("Memory L0 (observation) summary")
    print(f"vectors stored : {mem0_M.shape[0]} / dim {mem0_M.shape[1]}")
    if mem0_t.size:
        print(f"oldest         : {ts_to_str(float(mem0_t.min()))}")
        print(f"newest         : {ts_to_str(float(mem0_t.max()))}")

    mem1_M, mem1_t = data['mem1_M'], data['mem1_t']
    print_header("Memory L1 (sequence) summary")
    print(f"vectors stored : {mem1_M.shape[0]} / dim {mem1_M.shape[1]}")
    if mem1_t.size:
        print(f"oldest         : {ts_to_str(float(mem1_t.min()))}")
        print(f"newest         : {ts_to_str(float(mem1_t.max()))}")
    
    # ───── learning data ─────
    cell_exp = safe_get_dict(data, "cell_experience")
    if cell_exp:
        print_header("Cell Type Experience")
        
        # Convert to a table format
        cell_exp_table = []
        for cell_type, exp in cell_exp.items():
            reward = exp.get("reward", 0)
            visits = exp.get("visits", 0)
            reward_per_visit = reward / visits if visits > 0 else 0
            cell_exp_table.append([
                cell_type.capitalize(), 
                f"{reward:.2f}", 
                visits,
                f"{reward_per_visit:.2f}"
            ])
        
        print(tabulate(
            cell_exp_table,
            headers=["Cell Type", "Total Reward", "Visits", "Reward/Visit"],
            tablefmt="simple"
        ))
    
    # ───── history trends ─────
    history = safe_get_dict(data, "history")
    if history:
        print_header("Recent Performance Trends")
        history_analysis = analyze_history(history)
        
        if history_analysis:
            for metric in ["energy", "hunger", "pain", "food_stored"]:
                if metric in history_analysis:
                    stat = history_analysis[metric]
                    trend = stat.get("trend", "stable")
                    trend_symbol = "↗" if trend == "increasing" else "↘" if trend == "decreasing" else "→"
                    print(f"{metric.capitalize():<12}: {stat['last']:.1f} (avg: {stat['avg']:.1f}, min: {stat['min']:.1f}, max: {stat['max']:.1f}) {trend_symbol}")
            
            if "rewards" in history_analysis:
                reward_stat = history_analysis["rewards"]
                trend = reward_stat.get("trend", "stable")
                trend_symbol = "↗" if trend == "improving" else "↘" if trend == "worsening" else "→"
                print(f"{'Rewards':<12}: {reward_stat['last']:.1f} (avg: {reward_stat['avg']:.1f}) {trend_symbol}")
                print(f"Positive reward ratio: {reward_stat['positive_ratio']*100:.1f}%")
    
    # ───── detailed learning analysis ─────
    if args.learning:
        cell_exp = safe_get_dict(data, "cell_experience")
        q_values = safe_get_dict(data, "q_values")
        
        if cell_exp or q_values:
            print_header("Detailed Learning Analysis")
            
            learning_analysis = analyze_learning_data(cell_exp, q_values)
            
            print("Cell Type Value Assessment:")
            cell_values = [(cell, details["reward_per_visit"]) for cell, details in learning_analysis["cell_type_analysis"].items()]
            cell_values.sort(key=lambda x: x[1], reverse=True)
            for cell, value in cell_values:
                print(f"  {cell.capitalize():<8}: {value:.4f}")
            
            print("\nPreferred Actions:")
            action_prefs = learning_analysis["action_preferences"]
            total = sum(action_prefs.values()) or 1
            for action, count in sorted(action_prefs.items(), key=lambda x: x[1], reverse=True):
                if action:  # Action might be None
                    print(f"  {action:<5}: {count} ({count/total*100:.1f}%)")
            
            if "q_value_analysis" in learning_analysis and learning_analysis["q_value_analysis"]:
                q_stats = learning_analysis["q_value_analysis"]
                print("\nQ-Value Statistics:")
                print(f"  Count: {q_stats.get('count', 0)}")
                print(f"  Average: {q_stats.get('avg', 0):.4f}")
                print(f"  Range: [{q_stats.get('min', 0):.4f}, {q_stats.get('max', 0):.4f}]")
                print(f"  Std Dev: {q_stats.get('std', 0):.4f}")
        else:
            print("\nNo learning data available in this state file.")
    
    # ───── memory pattern analysis ─────
    if args.plot:
        print_header("Memory Pattern Analysis")
        mem0_stats = analyze_memory_patterns(mem0_M)
        mem1_stats = analyze_memory_patterns(mem1_M)
        
        print("Memory Layer 0 (Observation):")
        print(f"  Average pattern magnitude: {mem0_stats['avg_magnitude']:.4f}")
        print(f"  Pattern sparsity: {mem0_stats['avg_sparsity']*100:.1f}%")
        print(f"  Memory density: {mem0_stats['memory_density']:.4f}")
        
        print("\nMemory Layer 1 (Sequence):")
        print(f"  Average pattern magnitude: {mem1_stats['avg_magnitude']:.4f}")
        print(f"  Pattern sparsity: {mem1_stats['avg_sparsity']*100:.1f}%")
        print(f"  Memory density: {mem1_stats['memory_density']:.4f}")
        
        # Generate plots
        try:
            plot_memory_patterns(mem0_M, mem1_M)
        except Exception as e:
            print(f"Error generating memory visualizations: {e}")
            print("Try installing matplotlib with 'pip install matplotlib' if not already installed.")
    
    # ───── data export ─────
    if args.export:
        try:
            export_to_csv(data)
        except Exception as e:
            print(f"Error exporting data: {e}")
            print("Try installing pandas with 'pip install pandas' if not already installed.")
    
    # ───── optional full dump ─────
    if args.full:
        print_header("Full array dump")
        for key in data.files:
            print(f"{key}:\n" + indent(str(data[key]), "  ") + "\n")


if __name__ == "__main__":
    main()
