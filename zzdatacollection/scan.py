#!/usr/bin/env python3
"""
Scan.py - Speech-to-text accuracy analysis for all participants
Calculates WER, CER, and semantic similarity for Alzheimer's detection
"""

import os
import glob
from jiwer import wer, cer
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from datetime import datetime

def load_text_file(filepath):
    """Load text content from file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def scan_transcription(reference_text, transcription_file, output_dir):
    """
    Scan a single transcription against reference text.
    Calculates WER, CER, and semantic similarity.
    """
    # Load transcription
    hypothesis = load_text_file(transcription_file)
    if not hypothesis:
        return None
    
    # Get participant name from filename
    participant_name = os.path.splitext(os.path.basename(transcription_file))[0]
    
    # Load sentence transformer model
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    
    # Get embeddings for both texts
    reference_embedding = model.encode([reference_text])[0]
    hypothesis_embedding = model.encode([hypothesis])[0]
    
    # Calculate semantic similarity (0 to 1, where 1 is perfect match)
    similarity = 1 - cosine(reference_embedding, hypothesis_embedding)
    
    # Calculate WER and CER
    word_error_rate = wer(reference_text, hypothesis)
    char_error_rate = cer(reference_text, hypothesis)
    
    # Define threshold for assessment
    threshold = 0.65
    is_correct = similarity >= threshold
    
    # Create detailed result
    result = (
        f"Participant: {participant_name}\n"
        f"Reference: '{reference_text}'\n"
        f"Spoken: '{hypothesis}'\n"
        f"Word Error Rate (WER): {word_error_rate:.4f}\n"
        f"Character Error Rate (CER): {char_error_rate:.4f}\n"
        f"Semantic Similarity: {similarity:.4f}\n"
        f"Assessment: {'CORRECT' if is_correct else 'INCORRECT'}\n"
        f"Threshold: {threshold}\n"
    )
    
    # Save individual result
    output_file = os.path.join(output_dir, f"{participant_name}_scan.txt")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(result)
    
    print(f"✓ Scanned {participant_name}: WER={word_error_rate:.2%}, CER={char_error_rate:.2%}, Similarity={similarity:.2%}")
    
    return {
        'participant': participant_name,
        'wer': word_error_rate,
        'cer': char_error_rate,
        'similarity': similarity,
        'correct': is_correct
    }

def scan_all_transcriptions():
    """
    Scan all transcriptions in the transcriptions folder.
    """
    print("=" * 60)
    print("SPEECH ACCURACY ANALYSIS - SCAN ALL TRANSCRIPTIONS")
    print("=" * 60)
    
    # Setup directories (use parent directory paths)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    transcriptions_dir = os.path.join(base_dir, "transcriptions")
    reference_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "real.txt")
    output_dir = os.path.join(base_dir, "scan_results")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load reference text
    reference_text = load_text_file(reference_file)
    if not reference_text:
        print(f"Error: Could not load reference text from {reference_file}")
        return
    
    print(f"\nReference text loaded from {reference_file}")
    print(f"Reference: '{reference_text[:100]}...'\n")
    
    # Get all transcription files
    transcription_files = glob.glob(os.path.join(transcriptions_dir, "*.txt"))
    
    if not transcription_files:
        print(f"No transcription files found in {transcriptions_dir}/")
        return
    
    print(f"Found {len(transcription_files)} transcription files\n")
    
    # Process each transcription
    results = []
    for trans_file in transcription_files:
        result = scan_transcription(reference_text, trans_file, output_dir)
        if result:
            results.append(result)
    
    # Generate summary report
    if results:
        print("\n" + "=" * 60)
        print("SUMMARY REPORT")
        print("=" * 60)
        
        summary_file = os.path.join(output_dir, f"scan_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("SPEECH ACCURACY ANALYSIS SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Total participants scanned: {len(results)}\n")
            f.write(f"Reference text: {reference_text}\n\n")
            
            f.write("INDIVIDUAL RESULTS:\n")
            f.write("-" * 60 + "\n")
            
            for r in sorted(results, key=lambda x: x['wer']):
                f.write(f"\n{r['participant']}:\n")
                f.write(f"  WER: {r['wer']:.4f} ({r['wer']:.2%})\n")
                f.write(f"  CER: {r['cer']:.4f} ({r['cer']:.2%})\n")
                f.write(f"  Semantic Similarity: {r['similarity']:.4f} ({r['similarity']:.2%})\n")
                f.write(f"  Status: {'✓ CORRECT' if r['correct'] else '✗ INCORRECT'}\n")
            
            # Calculate statistics
            avg_wer = sum(r['wer'] for r in results) / len(results)
            avg_cer = sum(r['cer'] for r in results) / len(results)
            avg_sim = sum(r['similarity'] for r in results) / len(results)
            correct_count = sum(1 for r in results if r['correct'])
            
            f.write("\n" + "=" * 60 + "\n")
            f.write("STATISTICS:\n")
            f.write("-" * 60 + "\n")
            f.write(f"Average WER: {avg_wer:.4f} ({avg_wer:.2%})\n")
            f.write(f"Average CER: {avg_cer:.4f} ({avg_cer:.2%})\n")
            f.write(f"Average Semantic Similarity: {avg_sim:.4f} ({avg_sim:.2%})\n")
            f.write(f"Correct assessments: {correct_count}/{len(results)} ({correct_count/len(results):.2%})\n")
        
        print(f"\nResults saved to {output_dir}/")
        print(f"  - Individual scans: {len(results)} files")
        print(f"  - Summary report: {os.path.basename(summary_file)}")
        print(f"\nAverage WER: {avg_wer:.2%}")
        print(f"Average CER: {avg_cer:.2%}")
        print(f"Average Similarity: {avg_sim:.2%}")
        print(f"Success rate: {correct_count}/{len(results)} ({correct_count/len(results):.2%})")
    
    print("\n" + "=" * 60)
    print("SCAN COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    scan_all_transcriptions()
