from jiwer import wer, cer
from datetime import datetime
import glob
import os

# Find the most recent transcription file
transcriptions = glob.glob("transcription_*.txt")
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