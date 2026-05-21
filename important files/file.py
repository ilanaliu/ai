import sounddevice as sd
from scipy.io.wavfile import write
from datetime import datetime
import os

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