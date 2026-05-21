import os
import glob
import numpy as np
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pandas as pd
from scipy.stats import skew, kurtosis
from collections import defaultdict
import re
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def load_participant_data(base_dir):
    """Load participant age and health data from file."""
    participant_data = {}
    data_file = os.path.join(base_dir, "participant_data.txt")

    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                # Skip comments and empty lines
                if line.startswith('//') or not line.strip():
                    continue

                # Parse participant data
                name, age, health_status = line.strip().split(',')
                participant_data[name] = {
                    'age': int(age),
                    'health_status': health_status
                }
                print(f"Loaded data for participant: {name}")

    except FileNotFoundError:
        print(f"Warning: Participant data file not found at {data_file}")
    except Exception as e:
        print(f"Error loading participant data: {e}")

    return participant_data

def get_available_participants(recordings_dir):
    """Get list of available participants from recordings directory."""
    wav_files = glob.glob(os.path.join(recordings_dir, "*.wav"))
    participants = [os.path.splitext(os.path.basename(f))[0] for f in wav_files]
    return sorted(participants)

def select_participants(available_participants):
    """Allow user to select which participants to analyze."""
    print("\nAvailable participants:")
    print("0. Analyze ALL participants")
    for i, name in enumerate(available_participants, 1):
        print(f"{i}. {name}")

    print("\nSelect participants to analyze:")
    print("- Enter single number (e.g., '3') for one participant")
    print("- Enter multiple numbers separated by commas (e.g., '1,3,5') for multiple")
    print("- Enter '0' to analyze all participants")
    print("- Enter ranges with dash (e.g., '2-5') for consecutive participants")

    while True:
        try:
            selection = input("\nYour selection: ").strip()

            if selection == '0':
                return available_participants

            selected_indices = set()

            # Parse the selection
            parts = selection.split(',')
            for part in parts:
                part = part.strip()
                if '-' in part:
                    # Handle ranges like "2-5"
                    start, end = map(int, part.split('-'))
                    selected_indices.update(range(start, end + 1))
                else:
                    # Handle single numbers
                    selected_indices.add(int(part))

            # Convert indices to participant names
            selected_participants = []
            for idx in selected_indices:
                if 1 <= idx <= len(available_participants):
                    selected_participants.append(available_participants[idx - 1])
                else:
                    print(f"Warning: Index {idx} is out of range. Skipping.")

            if selected_participants:
                print(f"\nSelected participants: {', '.join(selected_participants)}")
                confirm = input("Confirm selection? (y/n): ").strip().lower()
                if confirm in ['y', 'yes']:
                    return selected_participants
                else:
                    print("Please make a new selection.")
            else:
                print("No valid participants selected. Please try again.")

        except ValueError:
            print("Invalid input format. Please use numbers, commas, and dashes only.")
        except Exception as e:
            print(f"Error processing selection: {e}")

def analyze_pauses(y, sr, silence_threshold=0.01):
    """Analyze pause patterns in speech."""
    # Calculate RMS energy
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]

    # Find silent regions
    silent_frames = rms < silence_threshold

    # Convert frames to time
    frame_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=512)

    # Find pause segments
    pause_starts = []
    pause_ends = []
    in_pause = False

    for i, is_silent in enumerate(silent_frames):
        if is_silent and not in_pause:
            pause_starts.append(frame_times[i])
            in_pause = True
        elif not is_silent and in_pause:
            pause_ends.append(frame_times[i])
            in_pause = False

    # Calculate pause statistics
    if len(pause_starts) > 0 and len(pause_ends) > 0:
        # Ensure equal lengths
        min_len = min(len(pause_starts), len(pause_ends))
        pause_durations = [pause_ends[i] - pause_starts[i] for i in range(min_len)]

        pause_stats = {
            'total_pauses': len(pause_durations),
            'avg_pause_duration': np.mean(pause_durations) if pause_durations else 0,
            'pause_frequency': len(pause_durations) / (len(y) / sr),  # pauses per second
            'longest_pause': max(pause_durations) if pause_durations else 0,
            'pause_variability': np.std(pause_durations) if pause_durations else 0
        }
    else:
        pause_stats = {
            'total_pauses': 0,
            'avg_pause_duration': 0,
            'pause_frequency': 0,
            'longest_pause': 0,
            'pause_variability': 0
        }

    return pause_stats

def analyze_transcription(transcription_file):
    """Analyze transcription text for linguistic patterns."""
    try:
        with open(transcription_file, 'r', encoding='utf-8') as f:
            text = f.read().strip()

        words = text.split()
        sentences = [s.strip() for s in text.split('.') if s.strip()]

        metrics = {
            'word_count': len(words),
            'avg_word_length': sum(len(w) for w in words) / max(len(words), 1),
            'sentence_count': len(sentences),
            'avg_sentence_length': len(words) / max(len(sentences), 1),
            'unique_words': len(set(words)),
            'vocabulary_richness': len(set(words)) / max(len(words), 1)
        }

        return metrics
    except Exception as e:
        print(f"Error analyzing transcription {transcription_file}: {e}")
        return None

def load_mfcc_data(mfcc_file):
    """Extract statistical features from MFCC data file."""
    features = {}
    try:
        with open(mfcc_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if 'Mean:' in line:
                    features['mean'] = float(line.split(':')[1])
                elif 'Std:' in line:
                    features['std'] = float(line.split(':')[1])
                elif 'Min:' in line:
                    features['min'] = float(line.split(':')[1])
                elif 'Max:' in line:
                    features['max'] = float(line.split(':')[1])
    except Exception as e:
        print(f"Error reading MFCC file {mfcc_file}: {e}")
    return features

def load_scan_results(scan_file):
    """Extract metrics from scan results."""
    metrics = defaultdict(float)
    try:
        with open(scan_file, 'r', encoding='utf-8') as f:
            content = f.read()

        patterns = {
            'wer': r'Word Error Rate.*?(\d+\.?\d*)%',
            'cer': r'Character Error Rate.*?(\d+\.?\d*)%',
            'similarity': r'Semantic Similarity.*?(\d+\.?\d*)%'
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, content)
            if match:
                metrics[key] = float(match.group(1)) / 100

        return dict(metrics)
    except Exception as e:
        print(f"Error reading scan file {scan_file}: {e}")
        return dict(metrics)

def analyze_recording(wav_file):
    """Extract acoustic features from recording."""
    try:
        y, sr = librosa.load(wav_file, sr=None)

        # Basic features
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        pitch, _ = librosa.piptrack(y=y, sr=sr)

        # Prosodic features
        zero_crossings = librosa.zero_crossings(y)
        zero_crossing_rate = float(sum(zero_crossings)) / len(zero_crossings)

        # Energy features
        rms = librosa.feature.rms(y=y)[0]
        energy_mean = float(np.mean(rms))
        energy_std = float(np.std(rms))

        # Rhythm features
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        pulse = librosa.beat.plp(onset_envelope=onset_env, sr=sr)
        rhythm_strength = float(np.mean(pulse))

        # Pause analysis
        pause_stats = analyze_pauses(y, sr)

        return {
            'tempo': float(tempo),
            'mfcc_stats': {
                'mean': float(np.mean(mfccs)),
                'std': float(np.std(mfccs)),
                'skewness': float(skew(mfccs.flatten())),
                'kurtosis': float(kurtosis(mfccs.flatten()))
            },
            'pitch_stats': {
                'mean': float(np.mean(pitch)),
                'std': float(np.std(pitch))
            },
            'prosodic_features': {
                'zero_crossing_rate': zero_crossing_rate,
                'energy_mean': energy_mean,
                'energy_std': energy_std,
                'rhythm_strength': rhythm_strength
            },
            'pause_analysis': pause_stats
        }
    except Exception as e:
        print(f"Error analyzing recording {wav_file}: {e}")
        return None

def calculate_risk_score(data, participant_info):
    """Calculate comprehensive risk score on a 0-50 scale."""
    risk_score = 0
    risk_factors = []

    # Age and Health Risk (12 points max)
    if participant_info:
        age = participant_info['age']
        if age >= 90:
            risk_factors.append("Age 90+ (very high risk)")
            risk_score += 12
        elif age >= 80:
            risk_factors.append("Age 80-89 (high risk)")
            risk_score += 10
        elif age >= 75:
            risk_factors.append("Age 75-79 (elevated risk)")
            risk_score += 8
        elif age >= 70:
            risk_factors.append("Age 70-74 (moderate risk)")
            risk_score += 5
        else:
            risk_factors.append("Age <70 (baseline risk)")
            risk_score += 2

        # Known diagnosis overrides everything
        if participant_info['health_status'] == 'early_alzheimers':
            risk_factors.append("CONFIRMED: Early Alzheimer's diagnosis")
            risk_score += 15  # Additional points for confirmed diagnosis

    # Language Pattern Analysis (15 points max)
    if 'scan_metrics' in data:
        sm = data['scan_metrics']
        # Word Error Rate (8 points)
        if sm.get('wer', 0) > 0.4:
            risk_factors.append(f"Severe word error rate ({sm['wer']:.1%})")
            risk_score += 8
        elif sm.get('wer', 0) > 0.3:
            risk_factors.append(f"High word error rate ({sm['wer']:.1%})")
            risk_score += 5
        elif sm.get('wer', 0) > 0.2:
            risk_factors.append(f"Moderate word error rate ({sm['wer']:.1%})")
            risk_score += 3

        # Semantic Coherence (7 points)
        if sm.get('similarity', 1.0) < 0.6:
            risk_factors.append(f"Very low semantic coherence ({sm['similarity']:.1%})")
            risk_score += 7
        elif sm.get('similarity', 1.0) < 0.7:
            risk_factors.append(f"Low semantic coherence ({sm['similarity']:.1%})")
            risk_score += 5
        elif sm.get('similarity', 1.0) < 0.8:
            risk_factors.append(f"Moderate semantic coherence ({sm['similarity']:.1%})")
            risk_score += 3

    # Acoustic Features (23 points max)
    if 'recording_features' in data:
        rf = data['recording_features']

        # Speech Rate (4 points)
        if rf['tempo'] < 80:
            risk_factors.append("Extremely slow speech rate (<80 BPM)")
            risk_score += 4
        elif rf['tempo'] < 90:
            risk_factors.append("Very slow speech rate (<90 BPM)")
            risk_score += 3
        elif rf['tempo'] < 100:
            risk_factors.append("Slow speech rate (<100 BPM)")
            risk_score += 2

        # Pitch Variation (4 points)
        if rf['pitch_stats']['std'] < 150:
            risk_factors.append("Extremely low pitch variation (<150)")
            risk_score += 4
        elif rf['pitch_stats']['std'] < 180:
            risk_factors.append("Very low pitch variation (<180)")
            risk_score += 3
        elif rf['pitch_stats']['std'] < 200:
            risk_factors.append("Low pitch variation (<200)")
            risk_score += 2

        # MFCC Variability (4 points)
        if rf['mfcc_stats']['std'] < 70:
            risk_factors.append("Very low MFCC variability (<70)")
            risk_score += 4
        elif rf['mfcc_stats']['std'] < 80:
            risk_factors.append("Low MFCC variability (<80)")
            risk_score += 2

        # Energy Variability (4 points)
        if rf['prosodic_features']['energy_std'] < 0.005:
            risk_factors.append("Extremely low energy variation (<0.005)")
            risk_score += 4
        elif rf['prosodic_features']['energy_std'] < 0.01:
            risk_factors.append("Very low energy variation (<0.01)")
            risk_score += 3
        elif rf['prosodic_features']['energy_std'] < 0.02:
            risk_factors.append("Low energy variation (<0.02)")
            risk_score += 1

        # Rhythm Strength (3 points)
        if rf['prosodic_features']['rhythm_strength'] < 0.1:
            risk_factors.append("Extremely weak rhythm patterns (<0.1)")
            risk_score += 3
        elif rf['prosodic_features']['rhythm_strength'] < 0.15:
            risk_factors.append("Very weak rhythm patterns (<0.15)")
            risk_score += 2
        elif rf['prosodic_features']['rhythm_strength'] < 0.2:
            risk_factors.append("Weak rhythm patterns (<0.2)")
            risk_score += 1

        # Pause Analysis (4 points)
        if 'pause_analysis' in rf:
            pa = rf['pause_analysis']
            if pa['avg_pause_duration'] > 2.0:
                risk_factors.append("Excessive pause duration (>2s average)")
                risk_score += 2
            elif pa['avg_pause_duration'] > 1.5:
                risk_factors.append("Long pause duration (>1.5s average)")
                risk_score += 1

            if pa['pause_frequency'] > 3.0:
                risk_factors.append("High pause frequency (>3 per second)")
                risk_score += 2
            elif pa['pause_frequency'] > 2.0:
                risk_factors.append("Elevated pause frequency (>2 per second)")
                risk_score += 1

    return risk_score, risk_factors

def get_diagnostic_assessment(risk_score, participant_info):
    """Provide definitive diagnostic assessment."""
    # Known diagnosis overrides analysis
    if participant_info and participant_info['health_status'] == 'early_alzheimers':
        return "YES - CONFIRMED ALZHEIMER'S", "Based on medical diagnosis"

    # Score-based assessment
    if risk_score >= 30:
        return "YES - LIKELY ALZHEIMER'S", "Multiple strong indicators suggest cognitive decline consistent with Alzheimer's"
    elif risk_score >= 22:
        return "POSSIBLE ALZHEIMER'S", "Several indicators present - requires professional evaluation"
    elif risk_score >= 15:
        return "MILD COGNITIVE CONCERNS", "Some patterns detected - monitor closely"
    else:
        return "NO - NORMAL PATTERNS", "Speech patterns within normal range for age"

def generate_individual_report(name, data, participant_info):
    """Generate detailed individual participant report."""
    report = []
    report.append(f"Participant: {name}")
    report.append("-" * 30)

    # Participant Information
    if participant_info:
        report.append(f"Age: {participant_info['age']}")
        report.append(f"Health Status: {participant_info['health_status']}\n")

    # Acoustic Analysis
    if 'recording_features' in data:
        rf = data['recording_features']
        report.append("ACOUSTIC SCORES")
        report.append(f"Speech Rate: {rf['tempo']:.2f} BPM")
        report.append(f"Pitch Variation: {rf['pitch_stats']['std']:.2f}")
        report.append(f"MFCC Score: {rf['mfcc_stats']['std']:.2f}")
        report.append(f"Energy Variation: {rf['prosodic_features']['energy_std']:.4f}")
        report.append(f"Rhythm Score: {rf['prosodic_features']['rhythm_strength']:.2f}")

        if 'pause_analysis' in rf:
            pa = rf['pause_analysis']
            report.append(f"Avg Pause Duration: {pa['avg_pause_duration']:.2f}s")
            report.append(f"Pause Frequency: {pa['pause_frequency']:.2f}/s")
        report.append("")

        # Additional detailed acoustic metrics
        report.append("DETAILED ACOUSTIC ANALYSIS")
        report.append(f"Pitch Mean: {rf['pitch_stats']['mean']:.2f}")
        report.append(f"MFCC Mean: {rf['mfcc_stats']['mean']:.2f}")
        report.append(f"MFCC Skewness: {rf['mfcc_stats']['skewness']:.2f}")
        report.append(f"MFCC Kurtosis: {rf['mfcc_stats']['kurtosis']:.2f}")
        report.append(f"Energy Mean: {rf['prosodic_features']['energy_mean']:.4f}")
        report.append(f"Zero Crossing Rate: {rf['prosodic_features']['zero_crossing_rate']:.4f}")

        if 'pause_analysis' in rf:
            pa = rf['pause_analysis']
            report.append(f"Total Pauses: {pa['total_pauses']}")
            report.append(f"Longest Pause: {pa['longest_pause']:.2f}s")
            report.append(f"Pause Variability: {pa['pause_variability']:.2f}")
        report.append("")

    # Language Analysis (if available)
    if 'scan_metrics' in data:
        sm = data['scan_metrics']
        report.append("LANGUAGE ANALYSIS")
        if sm.get('wer'):
            report.append(f"Word Error Rate: {sm['wer']:.1%}")
        if sm.get('cer'):
            report.append(f"Character Error Rate: {sm['cer']:.1%}")
        if sm.get('similarity'):
            report.append(f"Semantic Similarity: {sm['similarity']:.1%}")
        report.append("")

    # Transcription Analysis (if available)
    if 'transcription_metrics' in data:
        tm = data['transcription_metrics']
        report.append("TRANSCRIPTION ANALYSIS")
        report.append(f"Word Count: {tm['word_count']}")
        report.append(f"Average Word Length: {tm['avg_word_length']:.2f}")
        report.append(f"Sentence Count: {tm['sentence_count']}")
        report.append(f"Average Sentence Length: {tm['avg_sentence_length']:.2f}")
        report.append(f"Unique Words: {tm['unique_words']}")
        report.append(f"Vocabulary Richness: {tm['vocabulary_richness']:.2f}")
        report.append("")

    # Risk Assessment
    risk_score, risk_factors = calculate_risk_score(data, participant_info)
    report.append("RISK ASSESSMENT")
    report.append(f"Overall Score: {risk_score}/50")

    if risk_factors:
        report.append("\nRisk Factors Detected:")
        for factor in risk_factors:
            report.append(f"- {factor}")
    report.append("")

    # Diagnostic Assessment
    diagnosis, explanation = get_diagnostic_assessment(risk_score, participant_info)
    report.append(f"DIAGNOSTIC ASSESSMENT: {diagnosis}")
    report.append(f"Explanation: {explanation}\n")

    # Clinical Interpretation
    report.append("CLINICAL INTERPRETATION")
    if 'recording_features' in data:
        rf = data['recording_features']

        # Speech Rate Analysis
        if rf['tempo'] < 80:
            report.append("Speech Rate: Severely reduced - may indicate motor or cognitive impairment")
        elif rf['tempo'] < 100:
            report.append("Speech Rate: Below normal - possible mild cognitive decline")
        elif rf['tempo'] > 150:
            report.append("Speech Rate: Elevated - may indicate anxiety or compensatory behavior")
        else:
            report.append("Speech Rate: Within normal range")

        # Pitch Variation Analysis
        if rf['pitch_stats']['std'] < 150:
            report.append("Pitch Variation: Severely reduced prosody - strong indicator of cognitive decline")
        elif rf['pitch_stats']['std'] < 200:
            report.append("Pitch Variation: Reduced prosody - possible early cognitive changes")
        else:
            report.append("Pitch Variation: Normal prosodic range")

        # Energy Variation Analysis
        if rf['prosodic_features']['energy_std'] < 0.005:
            report.append("Energy Variation: Severely monotonic speech - concerning for dementia")
        elif rf['prosodic_features']['energy_std'] < 0.02:
            report.append("Energy Variation: Reduced vocal dynamics - mild concern")
        else:
            report.append("Energy Variation: Normal vocal dynamics")

        # Pause Analysis
        if 'pause_analysis' in rf:
            pa = rf['pause_analysis']
            if pa['avg_pause_duration'] > 2.0:
                report.append("Pause Patterns: Excessive hesitation - may indicate word-finding difficulties")
            elif pa['avg_pause_duration'] > 1.5:
                report.append("Pause Patterns: Elevated hesitation - mild language processing concern")
            else:
                report.append("Pause Patterns: Normal speech fluency")

    report.append("")

    # Recommendations
    report.append("RECOMMENDATION")
    if risk_score >= 35:
        report.append("URGENT - Immediate neurological evaluation required")
        report.append("Contact neurologist or geriatrician within 48 hours")
        report.append("Consider comprehensive cognitive assessment (MoCA, MMSE)")
        report.append("Brain imaging may be warranted")
    elif risk_score >= 25:
        report.append("HIGH PRIORITY - Professional cognitive assessment needed")
        report.append("Schedule appointment with healthcare provider within 2-4 weeks")
        report.append("Request formal cognitive screening")
        report.append("Consider referral to memory clinic")
    elif risk_score >= 15:
        report.append("MODERATE PRIORITY - Further evaluation suggested")
        report.append("Discuss findings with primary care physician")
        report.append("Annual cognitive screening recommended")
        report.append("Monitor for changes in daily functioning")
    else:
        report.append("LOW PRIORITY - Continued monitoring advised")
        report.append("Routine health monitoring")
        report.append("Repeat assessment in 12-24 months")
        report.append("Maintain healthy lifestyle practices")

    return "\n".join(report)

def generate_score_report(results, participant_data):
    """Generate a detailed score report in text format."""
    report = []
    report.append("ALZHEIMER'S DETECTION SCORE REPORT")
    report.append("=" * 50 + "\n")

    for name, data in results.items():
        report.append(f"Participant: {name}")
        report.append("-" * 30)

        # Add age and health info
        if name in participant_data:
            pd = participant_data[name]
            report.append(f"Age: {pd['age']}")
            report.append(f"Health Status: {pd['health_status']}\n")

        # Add acoustic scores
        if 'recording_features' in data:
            rf = data['recording_features']
            report.append("ACOUSTIC SCORES")
            report.append(f"Speech Rate: {rf['tempo']:.2f} BPM")
            report.append(f"Pitch Variation: {rf['pitch_stats']['std']:.2f}")
            report.append(f"MFCC Score: {rf['mfcc_stats']['std']:.2f}")
            report.append(f"Energy Variation: {rf['prosodic_features']['energy_std']:.4f}")
            report.append(f"Rhythm Score: {rf['prosodic_features']['rhythm_strength']:.2f}")

            if 'pause_analysis' in rf:
                pa = rf['pause_analysis']
                report.append(f"Avg Pause Duration: {pa['avg_pause_duration']:.2f}s")
                report.append(f"Pause Frequency: {pa['pause_frequency']:.2f}/s\n")

            # Add detailed acoustic metrics
            report.append("DETAILED ACOUSTIC ANALYSIS")
            report.append(f"Pitch Mean: {rf['pitch_stats']['mean']:.2f}")
            report.append(f"MFCC Mean: {rf['mfcc_stats']['mean']:.2f}")
            report.append(f"MFCC Skewness: {rf['mfcc_stats']['skewness']:.2f}")
            report.append(f"MFCC Kurtosis: {rf['mfcc_stats']['kurtosis']:.2f}")
            report.append(f"Energy Mean: {rf['prosodic_features']['energy_mean']:.4f}")
            report.append(f"Zero Crossing Rate: {rf['prosodic_features']['zero_crossing_rate']:.4f}")

            if 'pause_analysis' in rf:
                pa = rf['pause_analysis']
                report.append(f"Total Pauses: {pa['total_pauses']}")
                report.append(f"Longest Pause: {pa['longest_pause']:.2f}s")
                report.append(f"Pause Variability: {pa['pause_variability']:.2f}")
            report.append("")

        # Add language analysis if available
        if 'scan_metrics' in data:
            sm = data['scan_metrics']
            report.append("LANGUAGE ANALYSIS")
            if sm.get('wer'):
                report.append(f"Word Error Rate: {sm['wer']:.1%}")
            if sm.get('cer'):
                report.append(f"Character Error Rate: {sm['cer']:.1%}")
            if sm.get('similarity'):
                report.append(f"Semantic Similarity: {sm['similarity']:.1%}")
            report.append("")

        # Add transcription analysis if available
        if 'transcription_metrics' in data:
            tm = data['transcription_metrics']
            report.append("TRANSCRIPTION ANALYSIS")
            report.append(f"Word Count: {tm['word_count']}")
            report.append(f"Average Word Length: {tm['avg_word_length']:.2f}")
            report.append(f"Sentence Count: {tm['sentence_count']}")
            report.append(f"Average Sentence Length: {tm['avg_sentence_length']:.2f}")
            report.append(f"Unique Words: {tm['unique_words']}")
            report.append(f"Vocabulary Richness: {tm['vocabulary_richness']:.2f}")
            report.append("")

        # Add risk assessment
        risk_score, risk_factors = calculate_risk_score(data, participant_data.get(name))
        report.append("RISK ASSESSMENT")
        report.append(f"Overall Score: {risk_score}/50")

        if risk_factors:
            report.append("\nRisk Factors Detected:")
            for factor in risk_factors:
                report.append(f"- {factor}")

        # Add diagnostic assessment
        diagnosis, explanation = get_diagnostic_assessment(risk_score, participant_data.get(name))
        report.append(f"\nDIAGNOSTIC ASSESSMENT: {diagnosis}")
        report.append(f"Explanation: {explanation}")

        # Add clinical interpretation
        report.append("\nCLINICAL INTERPRETATION")
        if 'recording_features' in data:
            rf = data['recording_features']

            # Speech Rate Analysis
            if rf['tempo'] < 80:
                report.append("Speech Rate: Severely reduced - may indicate motor or cognitive impairment")
            elif rf['tempo'] < 100:
                report.append("Speech Rate: Below normal - possible mild cognitive decline")
            elif rf['tempo'] > 150:
                report.append("Speech Rate: Elevated - may indicate anxiety or compensatory behavior")
            else:
                report.append("Speech Rate: Within normal range")

            # Pitch Variation Analysis
            if rf['pitch_stats']['std'] < 150:
                report.append("Pitch Variation: Severely reduced prosody - strong indicator of cognitive decline")
            elif rf['pitch_stats']['std'] < 200:
                report.append("Pitch Variation: Reduced prosody - possible early cognitive changes")
            else:
                report.append("Pitch Variation: Normal prosodic range")

            # Energy Variation Analysis
            if rf['prosodic_features']['energy_std'] < 0.005:
                report.append("Energy Variation: Severely monotonic speech - concerning for dementia")
            elif rf['prosodic_features']['energy_std'] < 0.02:
                report.append("Energy Variation: Reduced vocal dynamics - mild concern")
            else:
                report.append("Energy Variation: Normal vocal dynamics")

            # Pause Analysis
            if 'pause_analysis' in rf:
                pa = rf['pause_analysis']
                if pa['avg_pause_duration'] > 2.0:
                    report.append("Pause Patterns: Excessive hesitation - may indicate word-finding difficulties")
                elif pa['avg_pause_duration'] > 1.5:
                    report.append("Pause Patterns: Elevated hesitation - mild language processing concern")
                else:
                    report.append("Pause Patterns: Normal speech fluency")

        # Add recommendation
        report.append("\nRECOMMENDATION")
        if risk_score >= 35:
            report.append("URGENT - Immediate neurological evaluation required")
            report.append("Contact neurologist or geriatrician within 48 hours")
            report.append("Consider comprehensive cognitive assessment (MoCA, MMSE)")
            report.append("Brain imaging may be warranted")
        elif risk_score >= 25:
            report.append("HIGH PRIORITY - Professional cognitive assessment needed")
            report.append("Schedule appointment with healthcare provider within 2-4 weeks")
            report.append("Request formal cognitive screening")
            report.append("Consider referral to memory clinic")
        elif risk_score >= 15:
            report.append("MODERATE PRIORITY - Further evaluation suggested")
            report.append("Discuss findings with primary care physician")
            report.append("Annual cognitive screening recommended")
            report.append("Monitor for changes in daily functioning")
        else:
            report.append("LOW PRIORITY - Continued monitoring advised")
            report.append("Routine health monitoring")
            report.append("Repeat assessment in 12-24 months")
            report.append("Maintain healthy lifestyle practices")

    return "\n".join(report)

def generate_score_report(results, participant_data):
    """Generate a detailed score report in text format."""
    report = []
    report.append("ALZHEIMER'S DETECTION SCORE REPORT")
    report.append("=" * 50 + "\n")

    for name, data in results.items():
        report.append(f"Participant: {name}")
        report.append("-" * 30)

        # Add age and health info
        if name in participant_data:
            pd = participant_data[name]
            report.append(f"Age: {pd['age']}")
            report.append(f"Health Status: {pd['health_status']}\n")

        # Add acoustic scores
        if 'recording_features' in data:
            rf = data['recording_features']
            report.append("ACOUSTIC SCORES")
            report.append(f"Speech Rate: {rf['tempo']:.2f} BPM")
            report.append(f"Pitch Variation: {rf['pitch_stats']['std']:.2f}")
            report.append(f"MFCC Score: {rf['mfcc_stats']['std']:.2f}")
            report.append(f"Energy Variation: {rf['prosodic_features']['energy_std']:.4f}")
            report.append(f"Rhythm Score: {rf['prosodic_features']['rhythm_strength']:.2f}")

            if 'pause_analysis' in rf:
                pa = rf['pause_analysis']
                report.append(f"Avg Pause Duration: {pa['avg_pause_duration']:.2f}s")
                report.append(f"Pause Frequency: {pa['pause_frequency']:.2f}/s\n")

            # Add detailed acoustic metrics
            report.append("DETAILED ACOUSTIC ANALYSIS")
            report.append(f"Pitch Mean: {rf['pitch_stats']['mean']:.2f}")
            report.append(f"MFCC Mean: {rf['mfcc_stats']['mean']:.2f}")
            report.append(f"MFCC Skewness: {rf['mfcc_stats']['skewness']:.2f}")
            report.append(f"MFCC Kurtosis: {rf['mfcc_stats']['kurtosis']:.2f}")
            report.append(f"Energy Mean: {rf['prosodic_features']['energy_mean']:.4f}")
            report.append(f"Zero Crossing Rate: {rf['prosodic_features']['zero_crossing_rate']:.4f}")

            if 'pause_analysis' in rf:
                pa = rf['pause_analysis']
                report.append(f"Total Pauses: {pa['total_pauses']}")
                report.append(f"Longest Pause: {pa['longest_pause']:.2f}s")
                report.append(f"Pause Variability: {pa['pause_variability']:.2f}")
            report.append("")

        # Add language analysis if available
        if 'scan_metrics' in data:
            sm = data['scan_metrics']
            report.append("LANGUAGE ANALYSIS")
            if sm.get('wer'):
                report.append(f"Word Error Rate: {sm['wer']:.1%}")
            if sm.get('cer'):
                report.append(f"Character Error Rate: {sm['cer']:.1%}")
            if sm.get('similarity'):
                report.append(f"Semantic Similarity: {sm['similarity']:.1%}")
            report.append("")

        # Add transcription analysis if available
        if 'transcription_metrics' in data:
            tm = data['transcription_metrics']
            report.append("TRANSCRIPTION ANALYSIS")
            report.append(f"Word Count: {tm['word_count']}")
            report.append(f"Average Word Length: {tm['avg_word_length']:.2f}")
            report.append(f"Sentence Count: {tm['sentence_count']}")
            report.append(f"Average Sentence Length: {tm['avg_sentence_length']:.2f}")
            report.append(f"Unique Words: {tm['unique_words']}")
            report.append(f"Vocabulary Richness: {tm['vocabulary_richness']:.2f}")
            report.append("")

        # Add risk assessment
        risk_score, risk_factors = calculate_risk_score(data, participant_data.get(name))
        report.append("RISK ASSESSMENT")
        report.append(f"Overall Score: {risk_score}/50")

        if risk_factors:
            report.append("\nRisk Factors Detected:")
            for factor in risk_factors:
                report.append(f"- {factor}")

        # Add diagnostic assessment
        diagnosis, explanation = get_diagnostic_assessment(risk_score, participant_data.get(name))
        report.append(f"\nDIAGNOSTIC ASSESSMENT: {diagnosis}")
        report.append(f"Explanation: {explanation}")

        # Add clinical interpretation
        report.append("\nCLINICAL INTERPRETATION")
        if 'recording_features' in data:
            rf = data['recording_features']

            # Speech Rate Analysis
            if rf['tempo'] < 80:
                report.append("Speech Rate: Severely reduced - may indicate motor or cognitive impairment")
            elif rf['tempo'] < 100:
                report.append("Speech Rate: Below normal - possible mild cognitive decline")
            elif rf['tempo'] > 150:
                report.append("Speech Rate: Elevated - may indicate anxiety or compensatory behavior")
            else:
                report.append("Speech Rate: Within normal range")

            # Pitch Variation Analysis
            if rf['pitch_stats']['std'] < 150:
                report.append("Pitch Variation: Severely reduced prosody - strong indicator of cognitive decline")
            elif rf['pitch_stats']['std'] < 200:
                report.append("Pitch Variation: Reduced prosody - possible early cognitive changes")
            else:
                report.append("Pitch Variation: Normal prosodic range")

            # Energy Variation Analysis
            if rf['prosodic_features']['energy_std'] < 0.005:
                report.append("Energy Variation: Severely monotonic speech - concerning for dementia")
            elif rf['prosodic_features']['energy_std'] < 0.02:
                report.append("Energy Variation: Reduced vocal dynamics - mild concern")
            else:
                report.append("Energy Variation: Normal vocal dynamics")

            # Pause Analysis
            if 'pause_analysis' in rf:
                pa = rf['pause_analysis']
                if pa['avg_pause_duration'] > 2.0:
                    report.append("Pause Patterns: Excessive hesitation - may indicate word-finding difficulties")
                elif pa['avg_pause_duration'] > 1.5:
                    report.append("Pause Patterns: Elevated hesitation - mild language processing concern")
                else:
                    report.append("Pause Patterns: Normal speech fluency")

        # Add recommendation
        report.append("\nRECOMMENDATION")
        if risk_score >= 35:
            report.append("URGENT - Immediate neurological evaluation required")
            report.append("Contact neurologist or geriatrician within 48 hours")
            report.append("Consider comprehensive cognitive assessment (MoCA, MMSE)")
            report.append("Brain imaging may be warranted")
        elif risk_score >= 25:
            report.append("HIGH PRIORITY - Professional cognitive assessment needed")
            report.append("Schedule appointment with healthcare provider within 2-4 weeks")
            report.append("Request formal cognitive screening")
            report.append("Consider referral to memory clinic")
        elif risk_score >= 15:
            report.append("MODERATE PRIORITY - Further evaluation suggested")
            report.append("Discuss findings with primary care physician")
            report.append("Annual cognitive screening recommended")
            report.append("Monitor for changes in daily functioning")
        else:
            report.append("LOW PRIORITY - Continued monitoring advised")
            report.append("Routine health monitoring")
            report.append("Repeat assessment in 12-24 months")
            report.append("Maintain healthy lifestyle practices")

        report.append("\n" + "=" * 50 + "\n")

    return "\n".join(report)

def analyze_all_files():
    """Perform comprehensive analysis of selected participants."""
    # Setup directories
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dirs = {
        'mfcc': os.path.join(base_dir, "mfcc_data"),
        'pitch': os.path.join(base_dir, "pitch_graphs"),
        'scan': os.path.join(base_dir, "scan_results"),
        'recordings': os.path.join(base_dir, "recordings"),
        'results': os.path.join(base_dir, "alzheimers_analysis"),
        'transcriptions': os.path.join(base_dir, "transcriptions"),
        'individual_reports': os.path.join(base_dir, "individual_reports")
    }

    os.makedirs(dirs['results'], exist_ok=True)
    os.makedirs(dirs['individual_reports'], exist_ok=True)

    # Validate recordings directory
    if not os.path.exists(dirs['recordings']):
        print(f"Error: Recordings directory not found at {dirs['recordings']}")
        return

    # Get available participants and let user select
    available_participants = get_available_participants(dirs['recordings'])
    if not available_participants:
        print("No WAV files found in recordings directory")
        return

    selected_participants = select_participants(available_participants)
    if not selected_participants:
        print("No participants selected. Exiting.")
        return

    # Load participant data
    participant_data = load_participant_data(base_dir)

    # Initialize results dictionary
    results = {}

    # Analyze each selected recording
    for participant in selected_participants:
        wav_file = os.path.join(dirs['recordings'], f"{participant}.wav")

        if not os.path.exists(wav_file):
            print(f"Warning: Recording file not found for {participant}")
            continue

        print(f"\nAnalyzing recording: {participant}")

        try:
            # Initialize data for this recording
            results[participant] = {}

            # Get recording features
            recording_features = analyze_recording(wav_file)
            if recording_features:
                results[participant]['recording_features'] = recording_features
                print("- Acoustic features analyzed")

            # Get MFCC features
            mfcc_file = os.path.join(dirs['mfcc'], f"{participant}_mfcc.txt")
            if os.path.exists(mfcc_file):
                results[participant]['mfcc_features'] = load_mfcc_data(mfcc_file)
                print("- MFCC data analyzed")

            # Get scan results
            scan_file = os.path.join(dirs['scan'], f"{participant}_scan.txt")
            if os.path.exists(scan_file):
                results[participant]['scan_metrics'] = load_scan_results(scan_file)
                print("- Scan results analyzed")

            # Get transcription analysis
            trans_file = os.path.join(dirs['transcriptions'], f"{participant}.txt")
            if os.path.exists(trans_file):
                results[participant]['transcription_metrics'] = analyze_transcription(trans_file)
                print("- Transcription analyzed")

            # Generate individual report
            individual_report = generate_individual_report(
                participant,
                results[participant],
                participant_data.get(participant)
            )
            individual_file = os.path.join(dirs['individual_reports'], f"{participant}_report.txt")
            with open(individual_file, 'w') as f:
                f.write(individual_report)
            print(f"- Individual report saved")

        except Exception as e:
            print(f"Error analyzing {participant}: {str(e)}")
            continue

    if not results:
        print("No participants were successfully analyzed.")
        return

    # Generate only the detailed score report (no summary)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Create filename suffix based on selection
    if len(selected_participants) == len(available_participants):
        suffix = "all"
    elif len(selected_participants) == 1:
        suffix = selected_participants[0]
    else:
        suffix = f"{len(selected_participants)}_selected"

    # Generate DETAILED score report only
    score_report = generate_score_report(results, participant_data)
    score_report_file = os.path.join(dirs['results'], f"detailed_score_report_{suffix}_{timestamp}.txt")

    with open(score_report_file, 'w') as f:
        f.write(score_report)

    print(f"\nAnalysis complete!")
    print(f"Analyzed {len(results)} participants: {', '.join(results.keys())}")
    print(f"DETAILED score report saved to: {score_report_file}")
    print(f"Individual reports saved to: {dirs['individual_reports']}")

if __name__ == "__main__":
    analyze_all_files()
