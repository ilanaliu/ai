import os
import glob
import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display

def select_recording(directory):
    """Let user select a WAV file from the directory."""
    wav_files = glob.glob(os.path.join(directory, "*.wav"))
    if not wav_files:
        return None

    print("\nAvailable recordings:")
    for idx, wav in enumerate(wav_files):
        print(f"{idx+1}: {os.path.basename(wav)}")

    choice = input("\nEnter the number of the recording to analyze (or press Enter for latest): ").strip()
    if choice.isdigit() and 1 <= int(choice) <= len(wav_files):
        return wav_files[int(choice)-1]
    return max(wav_files, key=os.path.getctime)

def extract_mfcc_from_file(wav_file, n_mfcc=40):
    """Extract MFCCs from a single WAV file."""
    y, sr = librosa.load(wav_file, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfccs, y, sr

def create_spectrogram(wav_file):
    """Create and save spectrogram for the audio file."""
    os.makedirs("spectrograms", exist_ok=True)

    y, sr = librosa.load(wav_file, sr=None)
    stft = librosa.stft(y)
    spectrogram = np.abs(stft)
    spectrogram_db = librosa.amplitude_to_db(spectrogram, ref=np.max)

    plt.figure(figsize=(12, 8))
    librosa.display.specshow(spectrogram_db,
                           sr=sr,
                           x_axis='time',
                           y_axis='hz')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Spectrogram - {os.path.basename(wav_file)}')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Frequency (Hz)')

    output_file = os.path.join("spectrograms",
                              f"{os.path.splitext(os.path.basename(wav_file))[0]}_spectrogram.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved spectrogram to: {output_file}")

def create_mel_spectrogram(wav_file):
    """Create and save mel spectrogram for the audio file."""
    os.makedirs("mel_spectrograms", exist_ok=True)

    y, sr = librosa.load(wav_file, sr=None)
    mel_spect = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_spect_db = librosa.power_to_db(mel_spect, ref=np.max)

    plt.figure(figsize=(12, 8))
    librosa.display.specshow(mel_spect_db,
                           sr=sr,
                           x_axis='time',
                           y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Mel Spectrogram - {os.path.basename(wav_file)}')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Mel Frequency')

    output_file = os.path.join("mel_spectrograms",
                              f"{os.path.splitext(os.path.basename(wav_file))[0]}_mel.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved mel spectrogram to: {output_file}")

def plot_mfcc(mfcc, filename):
    """Create detailed MFCC visualization."""
    os.makedirs("mfcc_plots", exist_ok=True)

    plt.figure(figsize=(14, 10))
    librosa.display.specshow(mfcc,
                           x_axis='time',
                           y_axis='mel',
                           cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'MFCC Analysis | 40 Coefficients | {os.path.basename(filename)}',
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    plt.ylabel('MFCC Coefficients', fontsize=12, fontweight='bold')

    tick_positions = range(0, mfcc.shape[0], 10)
    tick_labels = [f'MFCC-{i+1}' for i in tick_positions]
    plt.yticks(tick_positions, tick_labels, fontsize=10)
    plt.xticks(fontsize=11)
    plt.tight_layout(pad=3.0)

    output_file = os.path.join("mfcc_plots",
                              f"{os.path.splitext(os.path.basename(filename))[0]}_mfcc.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close()
    print(f"Saved MFCC plot to: {output_file}")

def save_mfcc_data(mfcc, filename, output_dir):
    """Save MFCC numerical data to text file."""
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(filename))[0]
    output_file = os.path.join(output_dir, f"{base_name}_mfcc.txt")

    with open(output_file, "w") as f:
        f.write(f"MFCC Analysis for {filename}\n")
        f.write("=" * 50 + "\n")
        f.write(f"Shape: {mfcc.shape}\n")
        f.write(f"Number of coefficients: {mfcc.shape[0]}\n")
        f.write(f"Number of frames: {mfcc.shape[1]}\n")
        f.write("=" * 50 + "\n\n")

        f.write("Statistical Summary:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Mean: {np.mean(mfcc):.4f}\n")
        f.write(f"Std: {np.std(mfcc):.4f}\n")
        f.write(f"Min: {np.min(mfcc):.4f}\n")
        f.write(f"Max: {np.max(mfcc):.4f}\n\n")

        f.write("Coefficient Values:\n")
        for i, row in enumerate(mfcc):
            f.write(f"\nCoefficient {i+1}:\n")
            np.savetxt(f, row, fmt='%.4f', delimiter=', ')

    print(f"Saved MFCC data to: {output_file}")

def analyze_audio():
    """Analyze selected audio file."""
    recordings_dir = "recordings"
    mfcc_data_dir = "mfcc_data"

    selected_file = select_recording(recordings_dir)
    if not selected_file:
        print("No WAV files found in the recordings directory.")
        return

    print(f"\nAnalyzing selected recording: {os.path.basename(selected_file)}")

    print("Extracting MFCCs...")
    mfcc, y, sr = extract_mfcc_from_file(selected_file)

    print("Generating MFCC plot...")
    plot_mfcc(mfcc, selected_file)

    print("Saving MFCC data...")
    save_mfcc_data(mfcc, selected_file, mfcc_data_dir)

    print("Generating spectrogram...")
    create_spectrogram(selected_file)

    print("Generating mel spectrogram...")
    create_mel_spectrogram(selected_file)

    print("\nAnalysis complete!")
    print("Created outputs for the selected recording:")
    print(f"- File analyzed: {os.path.basename(selected_file)}")
    print("- mfcc_plots: MFCC coefficient visualization")
    print("- mfcc_data: Numerical MFCC data")
    print("- spectrograms: Regular spectrogram")
    print("- mel_spectrograms: Mel-scale spectrogram")

if __name__ == "__main__":
    analyze_audio()
