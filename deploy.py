#!/usr/bin/env python3
"""
Alzheimer's Detection AI - Deployment Interface
==============================================
Simple launcher for using your trained AI model
"""

import os
import subprocess
import sys
import glob

def scan_all_files():
    """Automatically scan all audio files in common directories."""
    
    print("AUTO-SCANNING ALL AUDIO FILES")
    print("="*40)
    
    # Look for audio files in common directories
    search_dirs = [
        "recordings",
        "simulated_recordings", 
        ".",
        "audio_files"
    ]
    
    all_audio_files = []
    for search_dir in search_dirs:
        if os.path.exists(search_dir):
            wav_files = glob.glob(os.path.join(search_dir, "*.wav"))
            if wav_files:
                print(f"Found {len(wav_files)} files in {search_dir}/")
                all_audio_files.extend(wav_files)
    
    if not all_audio_files:
        print("No audio files (.wav) found in:")
        for d in search_dirs:
            print(f"   - {d}/")
        print("\nTip: Place .wav files in 'recordings/' or 'simulated_recordings/' folder")
        return
    
    print(f"\nTotal files found: {len(all_audio_files)}")
    print("Starting batch analysis...\n")
    
    # Create batch files list for the advanced_ensemble_bot
    batch_dir = "temp_batch_dir"
    os.makedirs(batch_dir, exist_ok=True)
    
    # Create symlinks or copy a few representative files
    sample_files = all_audio_files[:20] if len(all_audio_files) > 20 else all_audio_files
    
    for i, audio_file in enumerate(sample_files):
        dest = os.path.join(batch_dir, f"file_{i:03d}_{os.path.basename(audio_file)}")
        try:
            if os.name == 'nt':  # Windows
                import shutil
                shutil.copy2(audio_file, dest)
            else:  # Unix/Linux
                os.symlink(audio_file, dest)
        except:
            pass  # Skip if can't copy/link
    
    print(f"Processing {len(sample_files)} representative files...")
    print("Running batch analysis through main interface...\n")
    
    # Use the working deployment method
    try:
        result = subprocess.run([
            sys.executable, "bot.py", "deploy"
        ], input=f"2\n{batch_dir}\n3\n", text=True, capture_output=False)
    except Exception as e:
        print(f"Error running batch analysis: {e}")
    finally:
        # Cleanup
        try:
            import shutil
            shutil.rmtree(batch_dir)
        except:
            pass

def main():
    print("ALZHEIMER'S DETECTION AI - DEPLOYMENT")
    print("="*50)
    
    # Check if model exists
    if not os.path.exists("bot.pkl"):
        print("No trained model found!")
        print("   Please train the model first by running:")
        print("   python bot.py")
        input("\nPress Enter to exit...")
        return
    
    print("Trained model found!")
    print("\nDEPLOYMENT OPTIONS:")
    print("1. Scan ALL audio files automatically")
    print("2. Interactive mode (single/batch)")
    print("3. Exit")
    
    choice = input("\nSelect option (1-3): ").strip()
    
    if choice == "1":
        scan_all_files()
        
    elif choice == "2":
        print("Starting interactive deployment...\n")
        try:
            subprocess.run([sys.executable, "bot.py", "deploy"])
        except KeyboardInterrupt:
            print("\nDeployment stopped")
            
    elif choice == "3":
        print("Goodbye!")
        
    else:
        print("Invalid option!")

if __name__ == "__main__":
    main()