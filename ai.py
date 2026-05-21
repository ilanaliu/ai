import sounddevice as sd
from scipy.io.wavfile import write
from datetime import datetime
import glob
import os
import whisper
from jiwer import wer, cer

fs = 160000
seconds = 15

recording_folder = "recordings"
os.makedirs(recording_folder, exist_ok=True)

print("Recording...")
audio = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype='int16')
sd.wait()

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"recording_{timestamp}.wav"
filepath = os.path.join(recording_folder, filename)
write(filepath, fs, audio)
print(f"Recording saved as {filepath}")

recording_folder = "recordings"
recordings = glob.glob(os.path.join(recording_folder, "recording_*.wav"))
if not recordings:
    print("No recording files found.")
    exit()

latest_recording = max(recordings, key=os.path.getctime)
print(f"Transcribing {latest_recording}...")

model = whisper.load_model("large-v2")
result = model.transcribe(latest_recording)


transcription_folder = "transcriptions"
os.makedirs(transcription_folder, exist_ok=True)
timestamp = latest_recording.split("_")[-1].replace(".wav", "")
transcription_filename = f"transcription_{timestamp}.txt"
transcription_filepath = os.path.join(transcription_folder, transcription_filename)
with open(transcription_filepath, "w", encoding="utf-8") as f:
    f.write(result["text"])
print(f"Transcription saved as {transcription_filepath}")


transcriptions = glob.glob(os.path.join(transcription_folder, "transcription_*.txt"))
if not transcriptions:
    print("No transcription files found.")
    exit()

latest_transcription = max(transcriptions, key=os.path.getctime)
print(f"Scanning {latest_transcription}...")

with open(latest_transcription, "r", encoding="utf-8") as f:
    hypothesis = f.read().strip()

with open("real.txt", "r", encoding="utf-8") as f:
    reference = f.read().strip()

if not reference:
    print("Error: 'real.txt' is empty.")
elif not hypothesis:
    print(f"Error: '{latest_transcription}' is empty.")
else:
    word_error = wer(reference, hypothesis)
    char_error = cer(reference, hypothesis)

    result = (
        f"reference: '{reference}'\n"
        f"spoken: '{hypothesis}'\n"
        f"Word Error Rate (WER): {word_error:.2%}\n"
        f"Character Error Rate (CER): {char_error:.2%}\n"
    )


    scan_folder = "scan_results"
    os.makedirs(scan_folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"scan_result_{timestamp}.txt"
    filepath = os.path.join(scan_folder, filename)
    with open(filepath, "w", encoding="utf-8") as out:
        out.write(result)
    print(f"Results saved to {filepath}")