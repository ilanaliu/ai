import sounddevice as sd
from scipy.io.wavfile import write
from datetime import datetime
import threading
import glob
import os
import whisper
import numpy as np
from jiwer import wer, cer
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine

#record

fs = 16000
seconds = 300

recording_folder = "recordings"
os.makedirs(recording_folder, exist_ok=True)
def stop_recording():
    input("Press Enter to stop recording...")
    sd.stop()
print("Recording...Press Enter to stop.")
thread = threading.Thread(target=stop_recording, daemon=True)
thread.start()
audio = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype='int16')
sd.wait()
nonzero = np.flatnonzero(np.abs(audio) > 0)

if len(nonzero) > 0:
    actual_frames = nonzero[-1] + 1
else:
    actual_frames = 0
audio_trimmed = audio[:actual_frames]

duration_seconds = len(audio_trimmed) / fs
print(f"Recording duration: {duration_seconds:.2f} seconds")

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename = f"recording_{timestamp}.wav"
filepath = os.path.join(recording_folder, filename)
write(filepath, fs, audio_trimmed)
print(f"Recording saved as {filepath}")

#transcribe

recordings = glob.glob(os.path.join(recording_folder, "recording_*.wav"))
if not recordings:
    print("No recording files found.")
    exit()

latest_recording = max(recordings, key=os.path.getctime)
print(f"Transcribing {latest_recording}...")

model = whisper.load_model("large-v2")
result = model.transcribe(latest_recording, fp16=False)


transcription_folder = "transcriptions"
os.makedirs(transcription_folder, exist_ok=True)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
transcription_filename = f"transcription_{timestamp}.txt"

transcription_filepath = os.path.join(transcription_folder, transcription_filename)
with open(transcription_filepath, "w", encoding="utf-8") as f:
    f.write(result['text'] + "\n")
print(f"Transcription saved as {transcription_filepath}")

#scan

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
    # Load the sentence transformer model
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    # Get embeddings for both texts
    reference_embedding = model.encode([reference])[0]
    hypothesis_embedding = model.encode([hypothesis])[0]

    # Calculate semantic similarity (0 to 1, where 1 is perfect match)
    similarity = 1 - cosine(reference_embedding, hypothesis_embedding)

    # Define threshold for "close enough" (adjust as needed)
    threshold = 0.65
    is_correct = similarity >= threshold

    result = (
        f"Recording duration: {duration_seconds:.2f} seconds\n"
        f"Reference: '{reference}'\n"
        f"Spoken: '{hypothesis}'\n"
        f"Semantic Similarity: {similarity:.2%}\n"
        f"Assessment: {'CORRECT' if is_correct else 'INCORRECT'}\n"
        f"Word Error Rate (WER): {wer(reference, hypothesis):.2%}\n"
        f"Character Error Rate (CER): {cer(reference, hypothesis):.2%}\n"
    )

    scan_folder = "scan_results"
    os.makedirs(scan_folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"scan_result_{timestamp}.txt"
    filepath = os.path.join(scan_folder, filename)
    with open(filepath, "w", encoding="utf-8") as out:
        out.write(result)
    print(f"Results saved to {filepath}")
