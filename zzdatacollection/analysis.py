import os
import glob
import numpy as np
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def load_mfcc_data(filepath):
    """Load MFCC data from text file and return mean values."""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('Mean:'):
                return float(line.split(':')[1])
    return None

def load_transcription(filepath):
    """Load transcription text from file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read().strip()

def extract_features(wav_file):
    """Extract audio features from WAV file."""
    y, sr = librosa.load(wav_file, sr=None)

    # Extract features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    pitch, _ = librosa.piptrack(y=y, sr=sr)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)

    # Calculate statistics
    return {
        'mfcc_mean': np.mean(mfccs),
        'pitch_mean': np.mean(pitch),
        'mel_mean': np.mean(mel_spec),
        'duration': len(y) / sr
    }

def analyze_all():
    """Perform comprehensive analysis of all files."""
    print("Starting comprehensive analysis...")

    # Initialize paths (use parent directory)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    recordings_dir = os.path.join(base_dir, "recordings")
    transcriptions_dir = os.path.join(base_dir, "transcriptions")
    scan_results_dir = os.path.join(base_dir, "scan_results")

    # Load sentence transformer model
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    # Get all files
    wav_files = glob.glob(os.path.join(recordings_dir, "*.wav"))
    transcription_files = glob.glob(os.path.join(transcriptions_dir, "*.txt"))
    scan_files = glob.glob(os.path.join(scan_results_dir, "*.txt"))

    if not wav_files:
        print("No recordings found!")
        return

    # Create analysis directory
    analysis_dir = os.path.join(base_dir, "analysis_results")
    os.makedirs(analysis_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Prepare data structures
    features_dict = {}
    transcription_embeddings = {}

    print("\nExtracting features from recordings...")
    for wav_file in wav_files:
        basename = os.path.splitext(os.path.basename(wav_file))[0]
        features_dict[basename] = extract_features(wav_file)

    print("Processing transcriptions...")
    for trans_file in transcription_files:
        basename = os.path.splitext(os.path.basename(trans_file))[0]
        text = load_transcription(trans_file)
        transcription_embeddings[basename] = model.encode([text])[0]

    # Create feature matrix for clustering
    feature_matrix = []
    file_names = []

    for name, features in features_dict.items():
        # Only include participants with both audio features and transcriptions
        if name in transcription_embeddings:
            feature_vector = [
                features['mfcc_mean'],
                features['pitch_mean'],
                features['mel_mean'],
                features['duration']
            ]
            feature_vector.extend(transcription_embeddings[name].tolist())
            feature_matrix.append(feature_vector)
            file_names.append(name)

    # Normalize features
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(feature_matrix)

    # Perform PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(normalized_features)

    # Perform clustering
    n_clusters = min(3, len(feature_matrix))
    kmeans = KMeans(n_clusters=n_clusters)
    clusters = kmeans.fit_predict(normalized_features)

    # Create visualization
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1],
                         c=clusters, cmap='viridis')
    plt.title('Recording Clusters based on All Features')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')

    for i, name in enumerate(file_names):
        plt.annotate(name, (pca_result[i, 0], pca_result[i, 1]))

    plt.colorbar(scatter)

    # Save results
    analysis_file = os.path.join(analysis_dir, f"analysis_report_{timestamp}.txt")
    plot_file = os.path.join(analysis_dir, f"analysis_plot_{timestamp}.png")

    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()

    # Generate report
    with open(analysis_file, 'w') as f:
        f.write("Comprehensive Analysis Report\n")
        f.write("=" * 30 + "\n\n")

        f.write("Files Analyzed:\n")
        f.write("-" * 20 + "\n")
        for name in file_names:
            f.write(f"{name}\n")
            f.write(f"Cluster: {clusters[file_names.index(name)]}\n")
            f.write(f"Features: {features_dict[name]}\n\n")

        f.write("\nSimilarity Analysis:\n")
        f.write("-" * 20 + "\n")
        for i, name1 in enumerate(file_names):
            for j, name2 in enumerate(file_names):
                if i < j:
                    similarity = 1 - cosine(normalized_features[i], normalized_features[j])
                    f.write(f"{name1} vs {name2}: {similarity:.2%} similar\n")

    print("\nAnalysis complete!")
    print(f"Number of files analyzed: {len(file_names)}")
    print(f"\nResults saved in '{analysis_dir}':")
    print(f"- Plot: {os.path.basename(plot_file)}")
    print(f"- Report: {os.path.basename(analysis_file)}")

if __name__ == "__main__":
    analyze_all()
