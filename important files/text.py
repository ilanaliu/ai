import whisper
from datetime import datetime
import glob
import os

# Look for recordings in the 'recordings' folder
recording_folder = "recordings"
recordings = glob.glob(os.path.join(recording_folder, "recording_*.wav"))
if not recordings:
    print("No recording files found.")
    exit()

latest_recording = max(recordings, key=os.path.getctime)
print(f"Transcribing {latest_recording}...")

model = whisper.load_model("large-v2")
result = model.transcribe(latest_recording)

# Save transcription into a 'transcriptions' folder
transcription_folder = "transcriptions"
os.makedirs(transcription_folder, exist_ok=True)
timestamp = latest_recording.split("_")[-1].replace(".wav", "")
transcription_filename = f"transcription_{timestamp}.txt"
transcription_filepath = os.path.join(transcription_folder, transcription_filename)
with open(transcription_filepath, "w", encoding="utf-8") as f:
    f.write(result["text"])
print(f"Transcription saved as {transcription_filepath}")