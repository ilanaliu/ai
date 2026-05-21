import os
import glob
import librosa
import numpy as np
import matplotlib.pyplot as plt

def plot_pitch_contour(directory):
    # Get list of WAV files
    wav_files = glob.glob(os.path.join(directory, "*.wav"))
    if not wav_files:
        print("No WAV files found in 'recordings' folder.")
        return

    # Display available recordings
    print("\nAvailable recordings:")
    for idx, wav in enumerate(wav_files):
        print(f"{idx+1}: {os.path.basename(wav)}")

    # Get user selection
    choice = input("\nEnter the number of the recording to analyze (or press Enter for latest): ").strip()
    if choice.isdigit() and 1 <= int(choice) <= len(wav_files):
        selected_wav = wav_files[int(choice)-1]
    else:
        selected_wav = max(wav_files, key=os.path.getctime)

    print(f"\nAnalyzing {os.path.basename(selected_wav)}...")

    # Create output directory
    os.makedirs('pitch_graphs', exist_ok=True)

    # Process the selected file
    y, sr = librosa.load(selected_wav, sr=None)

    f0, voiced_flag, voiced_probs = librosa.pyin(y,
                                                fmin=librosa.note_to_hz('C2'),
                                                fmax=librosa.note_to_hz('C7'))
    times = librosa.times_like(f0)

    plt.figure(figsize=(12, 6))
    plt.plot(times[voiced_flag], f0[voiced_flag], 'b.', alpha=0.5, label='Pitch')
    plt.grid(True)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title(f'Pitch Contour - {os.path.basename(selected_wav)}')
    plt.legend()

    output_path = os.path.join('pitch_graphs',
                              f"{os.path.splitext(os.path.basename(selected_wav))[0]}_pitch.png")
    plt.savefig(output_path)
    plt.close()
    print(f"\nSaved pitch contour graph to: {output_path}")

if __name__ == "__main__":
    recordings_dir = "recordings"
    plot_pitch_contour(recordings_dir)
    print("\nProcessing complete!")
