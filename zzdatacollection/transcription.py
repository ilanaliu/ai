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

recording_folder = "recordings"
recordings = glob.glob(os.path.join(recording_folder, "*.wav"))
if not recordings:
    print("No recording files found.")
    exit()

print("Available recordings:")
for idx, rec in enumerate(recordings):
    print(f"{idx+1}: {os.path.basename(rec)}")

choice = input("Enter the number of the recording you want to transcribe (or press Enter for latest): ").strip()
if choice.isdigit() and 1 <= int(choice) <= len(recordings):
    selected_recording = recordings[int(choice)-1]
else:
    selected_recording = max(recordings, key=os.path.getctime)

print(f"Transcribing {selected_recording}...")

model = whisper.load_model("large-v2")
result = model.transcribe(selected_recording, fp16=False)


transcription_folder = "transcriptions"
os.makedirs(transcription_folder, exist_ok=True)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
transcription_filename = f"transcription_{timestamp}.txt"

transcription_filepath = os.path.join(transcription_folder, transcription_filename)
with open(transcription_filepath, "w", encoding="utf-8") as f:
    f.write(result['text'] + "\n")
print(f"Transcription saved as {transcription_filepath}")
