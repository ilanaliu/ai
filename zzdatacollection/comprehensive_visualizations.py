#!/usr/bin/env python3
"""
Comprehensive Data Visualization for Alzheimer's Detection Analysis
Combines scan results, participant data, and health status for insights
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def load_participant_data(filepath):
    """Load participant health data"""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('Participant'):
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 3:
                    name = parts[0]
                    try:
                        age = int(parts[1])
                    except:
                        age = 0
                    status = parts[2]
                    data.append({'name': name, 'age': age, 'health_status': status})
    return pd.DataFrame(data)

def load_scan_results(scan_dir):
    """Load all scan results"""
    scan_files = glob.glob(os.path.join(scan_dir, "*_scan.txt"))
    
    results = []
    for scan_file in scan_files:
        participant_name = os.path.basename(scan_file).replace('_scan.txt', '')
        
        with open(scan_file, 'r') as f:
            content = f.read()
            
        # Extract metrics
        wer = cer = similarity = None
        for line in content.split('\n'):
            if 'Word Error Rate' in line:
                wer = float(line.split(':')[1].strip())
            elif 'Character Error Rate' in line:
                cer = float(line.split(':')[1].strip())
            elif 'Semantic Similarity' in line:
                similarity = float(line.split(':')[1].strip())
        
        results.append({
            'name': participant_name,
            'wer': wer,
            'cer': cer,
            'similarity': similarity
        })
    
    return pd.DataFrame(results)

def create_comprehensive_visualizations():
    """Generate comprehensive visualizations"""
    
    print("="*70)
    print("COMPREHENSIVE DATA VISUALIZATION")
    print("="*70)
    
    # Setup paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    participant_file = os.path.join(base_dir, "zzdatacollection", "participant_data.txt")
    scan_dir = os.path.join(base_dir, "scan_results")
    output_dir = os.path.join(base_dir, "comprehensive_analysis")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print("\nLoading data...")
    participants = load_participant_data(participant_file)
    scan_results = load_scan_results(scan_dir)
    
    # Merge datasets
    df = pd.merge(participants, scan_results, on='name', how='inner')
    
    print(f"Loaded data for {len(df)} participants")
    print(f"Health status distribution:")
    print(df['health_status'].value_counts())
    
    # Configure plotting style
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    
    # ===================================================================
    # GRAPH 1: WER and CER by Health Status
    # ===================================================================
    print("\nGenerating Graph 1: Error Rates by Health Status...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # WER by health status
    health_order = ['no_history', 'cognitive_decline', 'early_alzheimers']
    df_sorted = df.sort_values('health_status', key=lambda x: x.map({s: i for i, s in enumerate(health_order)}))
    
    sns.boxplot(data=df_sorted, x='health_status', y='wer', ax=ax1)
    sns.stripplot(data=df_sorted, x='health_status', y='wer', color='black', alpha=0.5, ax=ax1)
    ax1.set_title('Word Error Rate by Health Status', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Health Status', fontsize=12)
    ax1.set_ylabel('Word Error Rate', fontsize=12)
    ax1.set_xticklabels(['No History', 'Cognitive Decline', 'Early Alzheimers'])
    
    # CER by health status
    sns.boxplot(data=df_sorted, x='health_status', y='cer', ax=ax2)
    sns.stripplot(data=df_sorted, x='health_status', y='cer', color='black', alpha=0.5, ax=ax2)
    ax2.set_title('Character Error Rate by Health Status', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Health Status', fontsize=12)
    ax2.set_ylabel('Character Error Rate', fontsize=12)
    ax2.set_xticklabels(['No History', 'Cognitive Decline', 'Early Alzheimers'])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '1_error_rates_by_health_status.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # ===================================================================
    # GRAPH 2: Semantic Similarity by Health Status
    # ===================================================================
    print("Generating Graph 2: Semantic Similarity by Health Status...")
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    sns.violinplot(data=df_sorted, x='health_status', y='similarity', ax=ax)
    sns.stripplot(data=df_sorted, x='health_status', y='similarity', color='red', alpha=0.6, size=8, ax=ax)
    
    ax.set_title('Semantic Similarity by Health Status', fontsize=16, fontweight='bold')
    ax.set_xlabel('Health Status', fontsize=12)
    ax.set_ylabel('Semantic Similarity Score', fontsize=12)
    ax.set_xticklabels(['No History', 'Cognitive Decline', 'Early Alzheimers'])
    ax.axhline(y=0.65, color='green', linestyle='--', label='Threshold (0.65)')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '2_semantic_similarity_by_health.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # ===================================================================
    # GRAPH 3: Age vs Biomarkers
    # ===================================================================
    print("Generating Graph 3: Age Correlations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # WER vs Age
    for health_status in df['health_status'].unique():
        subset = df[df['health_status'] == health_status]
        axes[0, 0].scatter(subset['age'], subset['wer'], label=health_status, s=100, alpha=0.7)
    axes[0, 0].set_xlabel('Age', fontsize=12)
    axes[0, 0].set_ylabel('Word Error Rate', fontsize=12)
    axes[0, 0].set_title('WER vs Age', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # CER vs Age
    for health_status in df['health_status'].unique():
        subset = df[df['health_status'] == health_status]
        axes[0, 1].scatter(subset['age'], subset['cer'], label=health_status, s=100, alpha=0.7)
    axes[0, 1].set_xlabel('Age', fontsize=12)
    axes[0, 1].set_ylabel('Character Error Rate', fontsize=12)
    axes[0, 1].set_title('CER vs Age', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Similarity vs Age
    for health_status in df['health_status'].unique():
        subset = df[df['health_status'] == health_status]
        axes[1, 0].scatter(subset['age'], subset['similarity'], label=health_status, s=100, alpha=0.7)
    axes[1, 0].set_xlabel('Age', fontsize=12)
    axes[1, 0].set_ylabel('Semantic Similarity', fontsize=12)
    axes[1, 0].set_title('Semantic Similarity vs Age', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(y=0.65, color='green', linestyle='--', alpha=0.5)
    
    # WER vs CER colored by health status
    for health_status in df['health_status'].unique():
        subset = df[df['health_status'] == health_status]
        axes[1, 1].scatter(subset['wer'], subset['cer'], label=health_status, s=100, alpha=0.7)
    axes[1, 1].set_xlabel('Word Error Rate', fontsize=12)
    axes[1, 1].set_ylabel('Character Error Rate', fontsize=12)
    axes[1, 1].set_title('WER vs CER by Health Status', fontsize=14, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '3_age_correlations.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # ===================================================================
    # GRAPH 4: Participant Performance Dashboard
    # ===================================================================
    print("Generating Graph 4: Participant Performance Dashboard...")
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 14))
    
    # Sort by WER
    df_sorted_perf = df.sort_values('wer')
    
    # WER bar chart
    colors = df_sorted_perf['health_status'].map({
        'no_history': 'green',
        'cognitive_decline': 'orange',
        'early_alzheimers': 'red'
    })
    
    axes[0].barh(df_sorted_perf['name'], df_sorted_perf['wer'], color=colors, alpha=0.7)
    axes[0].set_xlabel('Word Error Rate', fontsize=12)
    axes[0].set_title('Word Error Rate by Participant', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='x')
    
    # CER bar chart
    df_sorted_cer = df.sort_values('cer')
    colors_cer = df_sorted_cer['health_status'].map({
        'no_history': 'green',
        'cognitive_decline': 'orange',
        'early_alzheimers': 'red'
    })
    
    axes[1].barh(df_sorted_cer['name'], df_sorted_cer['cer'], color=colors_cer, alpha=0.7)
    axes[1].set_xlabel('Character Error Rate', fontsize=12)
    axes[1].set_title('Character Error Rate by Participant', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='x')
    
    # Similarity bar chart
    df_sorted_sim = df.sort_values('similarity', ascending=False)
    colors_sim = df_sorted_sim['health_status'].map({
        'no_history': 'green',
        'cognitive_decline': 'orange',
        'early_alzheimers': 'red'
    })
    
    axes[2].barh(df_sorted_sim['name'], df_sorted_sim['similarity'], color=colors_sim, alpha=0.7)
    axes[2].set_xlabel('Semantic Similarity', fontsize=12)
    axes[2].set_title('Semantic Similarity by Participant', fontsize=14, fontweight='bold')
    axes[2].axvline(x=0.65, color='black', linestyle='--', linewidth=2, label='Threshold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '4_participant_dashboard.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # ===================================================================
    # GRAPH 5: Statistics Summary Heatmap
    # ===================================================================
    print("Generating Graph 5: Statistics Summary Heatmap...")
    
    # Calculate statistics by health status
    stats_summary = df.groupby('health_status').agg({
        'wer': ['mean', 'std'],
        'cer': ['mean', 'std'],
        'similarity': ['mean', 'std'],
        'age': ['mean', 'std']
    }).round(3)
    
    # Create heatmap data
    heatmap_data = pd.DataFrame({
        'WER Mean': stats_summary[('wer', 'mean')],
        'WER Std': stats_summary[('wer', 'std')],
        'CER Mean': stats_summary[('cer', 'mean')],
        'CER Std': stats_summary[('cer', 'std')],
        'Similarity Mean': stats_summary[('similarity', 'mean')],
        'Similarity Std': stats_summary[('similarity', 'std')],
        'Age Mean': stats_summary[('age', 'mean')],
        'Age Std': stats_summary[('age', 'std')]
    })
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(heatmap_data.T, annot=True, fmt='.3f', cmap='RdYlGn_r', ax=ax, cbar_kws={'label': 'Value'})
    ax.set_title('Statistics Summary Heatmap by Health Status', fontsize=16, fontweight='bold')
    ax.set_xlabel('Health Status', fontsize=12)
    ax.set_ylabel('Metric', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '5_statistics_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # ===================================================================
    # Generate Summary Report
    # ===================================================================
    print("\nGenerating Summary Report...")
    
    report_file = os.path.join(output_dir, 'comprehensive_analysis_report.txt')
    
    with open(report_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("COMPREHENSIVE ALZHEIMER'S DETECTION ANALYSIS REPORT\n")
        f.write("="*70 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Participants: {len(df)}\n\n")
        
        f.write("HEALTH STATUS DISTRIBUTION:\n")
        f.write("-"*70 + "\n")
        for status, count in df['health_status'].value_counts().items():
            f.write(f"  {status}: {count} participants\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("STATISTICS BY HEALTH STATUS:\n")
        f.write("="*70 + "\n\n")
        
        for status in ['no_history', 'cognitive_decline', 'early_alzheimers']:
            subset = df[df['health_status'] == status]
            if len(subset) > 0:
                f.write(f"\n{status.upper().replace('_', ' ')}:\n")
                f.write("-"*70 + "\n")
                f.write(f"  Count: {len(subset)}\n")
                f.write(f"  Age: {subset['age'].mean():.1f} ± {subset['age'].std():.1f} years\n")
                f.write(f"  WER: {subset['wer'].mean():.4f} ± {subset['wer'].std():.4f}\n")
                f.write(f"  CER: {subset['cer'].mean():.4f} ± {subset['cer'].std():.4f}\n")
                f.write(f"  Semantic Similarity: {subset['similarity'].mean():.4f} ± {subset['similarity'].std():.4f}\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("KEY FINDINGS:\n")
        f.write("="*70 + "\n\n")
        
        # Calculate key findings
        avg_wer_healthy = df[df['health_status'] == 'no_history']['wer'].mean()
        avg_wer_ad = df[df['health_status'] == 'early_alzheimers']['wer'].mean() if len(df[df['health_status'] == 'early_alzheimers']) > 0 else 0
        
        f.write(f"1. Average WER (No History): {avg_wer_healthy:.2%}\n")
        if avg_wer_ad > 0:
            f.write(f"2. Average WER (Early Alzheimer's): {avg_wer_ad:.2%}\n")
            f.write(f"3. WER Difference: {(avg_wer_ad - avg_wer_healthy):.2%}\n")
        
        # Correlation analysis
        f.write(f"\n4. Correlation between Age and WER: {df['age'].corr(df['wer']):.3f}\n")
        f.write(f"5. Correlation between Age and Similarity: {df['age'].corr(df['similarity']):.3f}\n")
        
        # Participants below threshold
        below_threshold = df[df['similarity'] < 0.65]
        f.write(f"\n6. Participants below similarity threshold (0.65): {len(below_threshold)}/{len(df)}\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("VISUALIZATIONS GENERATED:\n")
        f.write("="*70 + "\n")
        f.write("  1. Error Rates by Health Status\n")
        f.write("  2. Semantic Similarity by Health Status\n")
        f.write("  3. Age Correlations\n")
        f.write("  4. Participant Performance Dashboard\n")
        f.write("  5. Statistics Summary Heatmap\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*70 + "\n")
    
    print("\n" + "="*70)
    print("COMPREHENSIVE ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - 1_error_rates_by_health_status.png")
    print("  - 2_semantic_similarity_by_health.png")
    print("  - 3_age_correlations.png")
    print("  - 4_participant_dashboard.png")
    print("  - 5_statistics_heatmap.png")
    print("  - comprehensive_analysis_report.txt")
    print("\n" + "="*70)

if __name__ == "__main__":
    create_comprehensive_visualizations()
