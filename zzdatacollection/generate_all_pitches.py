#!/usr/bin/env python3
"""
Generate pitch graphs for all participants
"""
import os
import glob
import librosa
import numpy as np
import matplotlib.pyplot as plt

def generate_all_pitch_graphs():
    """Generate pitch contour graphs for all recordings"""
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    recordings_dir = os.path.join(base_dir, "recordings")
    output_dir = os.path.join(base_dir, "pitch_graphs")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all WAV files
    wav_files = glob.glob(os.path.join(recordings_dir, "*.wav"))
    
    if not wav_files:
        print("No WAV files found in recordings folder.")
        return
    
    print(f"Generating pitch graphs for {len(wav_files)} recordings...")
    print("="*60)
    
    for idx, wav_file in enumerate(wav_files, 1):
        participant_name = os.path.splitext(os.path.basename(wav_file))[0]
        print(f"{idx}/{len(wav_files)}: Processing {participant_name}...")
        
        try:
            # Load audio file
            y, sr = librosa.load(wav_file, sr=None)
            
            # Extract pitch using pyin
            f0, voiced_flag, voiced_probs = librosa.pyin(
                y,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7')
            )
            
            times = librosa.times_like(f0)
            
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
            
            # Plot 1: Pitch contour
            ax1.plot(times[voiced_flag], f0[voiced_flag], 'b.', alpha=0.5, markersize=3)
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Frequency (Hz)')
            ax1.set_title(f'Pitch Contour - {participant_name}')
            ax1.grid(True, alpha=0.3)
            
            # Calculate statistics
            f0_clean = f0[voiced_flag]
            if len(f0_clean) > 0:
                mean_f0 = np.mean(f0_clean)
                std_f0 = np.std(f0_clean)
                min_f0 = np.min(f0_clean)
                max_f0 = np.max(f0_clean)
                
                # Add horizontal lines for mean
                ax1.axhline(y=mean_f0, color='r', linestyle='--', alpha=0.7, label=f'Mean: {mean_f0:.1f} Hz')
                ax1.legend()
                
                # Plot 2: Histogram
                ax2.hist(f0_clean, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
                ax2.axvline(x=mean_f0, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean_f0:.1f} Hz')
                ax2.set_xlabel('Frequency (Hz)')
                ax2.set_ylabel('Count')
                ax2.set_title('Pitch Distribution')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                # Add statistics text box
                stats_text = f'Mean: {mean_f0:.1f} Hz\nStd: {std_f0:.1f} Hz\nMin: {min_f0:.1f} Hz\nMax: {max_f0:.1f} Hz'
                ax2.text(0.98, 0.97, stats_text, transform=ax2.transAxes,
                        verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.tight_layout()
            
            # Save the plot
            output_path = os.path.join(output_dir, f"{participant_name}_pitch.png")
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  ✓ Saved: {participant_name}_pitch.png")
            
        except Exception as e:
            print(f"  ✗ Error processing {participant_name}: {e}")
    
    print("\n" + "="*60)
    print(f"Pitch graph generation complete!")
    print(f"Saved {len(wav_files)} graphs to: {output_dir}")

if __name__ == "__main__":
    generate_all_pitch_graphs()
