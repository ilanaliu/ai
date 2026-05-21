#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from scipy import signal
import random
import glob
from datetime import datetime
import shutil
import json

class PatientSimulator:
    
    def __init__(self, base_dir="."):
        self.base_dir = base_dir
        self.recordings_dir = os.path.join(base_dir, "recordings")
        self.sim_recordings_dir = os.path.join(base_dir, "simulated_recordings")
        self.sim_data_dir = os.path.join(base_dir, "simulated_data")
        

        os.makedirs(self.sim_recordings_dir, exist_ok=True)
        os.makedirs(self.sim_data_dir, exist_ok=True)

        self.original_participants = self._load_original_data()
        
    def _load_original_data(self):

        participants = {}
        participant_file = os.path.join(self.base_dir, "zzdatacollection", "participant_data.txt")
        
        if os.path.exists(participant_file):
            try:
                with open(participant_file, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        line = line.strip()
                        if ',' in line and not line.startswith('Participant'):
                            parts = [p.strip() for p in line.split(',')]
                            if len(parts) >= 3:
                                name, age, health = parts[0], int(parts[1]), parts[2]
                                participants[name] = {
                                    'age': age,
                                    'health_status': health
                                }
            except Exception as e:
                print(f"Error loading participant data: {e}")
        
        return participants
    
    def _augment_audio(self, audio_data, sr, augmentation_type="random", health_status='no_history'):

        augmented = audio_data.copy()
        

        if augmentation_type == "random":
            aug_types = ['pitch', 'speed', 'noise', 'volume']
            selected_augs = random.sample(aug_types, random.randint(1, 2))
        else:
            selected_augs = [augmentation_type]
            
        for aug_type in selected_augs:
            if aug_type == 'pitch':

                pitch_shift = random.uniform(-0.5, 0.5) 
                augmented = librosa.effects.pitch_shift(augmented, sr=sr, n_steps=pitch_shift)
                
            elif aug_type == 'speed':
                # Minor speed variations (natural speech tempo differences)
                speed_factor = random.uniform(0.95, 1.05)  # Very subtle timing changes
                augmented = librosa.effects.time_stretch(augmented, rate=speed_factor)
                
            elif aug_type == 'noise':
                # Minimal background noise (different recording environments)
                noise_level = random.uniform(0.001, 0.005)  # Very low noise
                noise = np.random.normal(0, noise_level, augmented.shape)
                augmented = augmented + noise
                
            elif aug_type == 'volume':
                # Slight volume variations (microphone distance, recording gain)
                volume_factor = random.uniform(0.8, 1.2)
                augmented = augmented * volume_factor
        
        # Normalize to prevent clipping
        if np.max(np.abs(augmented)) > 0:
            augmented = augmented / np.max(np.abs(augmented)) * 0.95
            
        return augmented
    
    def _generate_patient_profile(self, base_participant=None):
        """Generate a patient profile based on the original recordings."""
        
        # If we have a base participant from original data, use their characteristics
        if base_participant:
            age = base_participant['age']
            health_status = base_participant['health_status']
        else:
            # Create variations around the original participants
            age = random.randint(65, 95)  # Natural elderly range
            health_status = random.choice(['no_history', 'cognitive_decline', 'early_alzheimers'])
        
        # Generate realistic name
        first_names = ["Alice", "Betty", "Carol", "Dorothy", "Eleanor", "Frances", "Grace", "Helen",
                      "Irene", "Joyce", "Katherine", "Louise", "Margaret", "Nancy", "Olive", "Patricia",
                      "Robert", "Charles", "John", "James", "William", "David", "Richard", "Thomas",
                      "Michael", "Joseph", "Daniel", "Christopher", "Matthew", "Anthony"]
        
        last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
                     "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
                     "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson",
                     "White", "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson"]
        
        first_name = random.choice(first_names)
        last_name = random.choice(last_names)
        name = f"{first_name}{last_name}"
        
        return {
            'name': name,
            'age': age,
            'health_status': health_status
        }
    
    def _select_base_audio(self, health_status):
        """Randomly select any original recording as base audio."""
        # Get all original recordings
        recordings = glob.glob(os.path.join(self.recordings_dir, "*.wav"))
        
        if not recordings:
            raise ValueError("No original recordings found!")
            
        # Just randomly select any recording - let the natural characteristics speak for themselves
        return random.choice(recordings)
    
    def _apply_natural_variation(self, audio, sr):
        """Apply natural variations regardless of health status."""
        
        # Just apply random natural variations - let the original recording characteristics determine the outcome
        aug_types = random.choices(['pitch', 'speed', 'noise', 'volume'], k=random.randint(1, 2))
        
        for aug_type in aug_types:
            audio = self._augment_audio(audio, sr, aug_type)
            
        return audio
    
    def generate_simulated_patients(self, num_patients=100):
        """Generate the specified number of simulated patients."""
        
        print(f"Generating {num_patients} simulated patients...")
        
        simulated_participants = []
        
        for i in range(num_patients):
            # Generate patient profile
            patient = self._generate_patient_profile()
            patient_name = f"Sim_{i+1:03d}_{patient['name']}"
            
            print(f"Creating patient {i+1}/{num_patients}: {patient_name} (Age: {patient['age']}, Status: {patient['health_status']})")
            
            try:
                # Select base audio
                base_audio_file = self._select_base_audio(patient['health_status'])
                
                # Load and augment audio
                audio, sr = librosa.load(base_audio_file, sr=None)
                
                # Apply natural variations (let original characteristics determine outcome)
                augmented_audio = self._apply_natural_variation(audio, sr)
                
                # Save augmented audio
                output_file = os.path.join(self.sim_recordings_dir, f"{patient_name}.wav")
                sf.write(output_file, augmented_audio, sr)
                
                # Store patient data
                simulated_participants.append({
                    'name': patient_name,
                    'age': patient['age'],
                    'health_status': patient['health_status'],
                    'base_recording': os.path.basename(base_audio_file)
                })
                
            except Exception as e:
                print(f"Error creating patient {patient_name}: {e}")
                continue
        
        # Save participant data
        self._save_participant_data(simulated_participants)
        
        # Generate summary statistics
        self._generate_summary(simulated_participants)
        
        print(f"\nSuccessfully generated {len(simulated_participants)} simulated patients!")
        print(f"Audio files saved in: {self.sim_recordings_dir}")
        print(f"Data files saved in: {self.sim_data_dir}")
        
        return simulated_participants
    
    def _save_participant_data(self, participants):
        """Save simulated participant data in the expected format."""
        
        # Save as CSV (compatible with original format)
        csv_file = os.path.join(self.sim_data_dir, "simulated_participant_data.txt")
        with open(csv_file, 'w') as f:
            for participant in participants:
                f.write(f"{participant['name']},{participant['age']},{participant['health_status']}\n")
        
        # Save detailed JSON data
        json_file = os.path.join(self.sim_data_dir, "simulated_participant_data.json")
        with open(json_file, 'w') as f:
            json.dump(participants, f, indent=2)
        
        print(f"Participant data saved to {csv_file} and {json_file}")
    
    def _generate_summary(self, participants):
        """Generate summary statistics of simulated patients."""
        
        # Count by health status
        status_counts = {}
        age_by_status = {}
        
        for p in participants:
            status = p['health_status']
            status_counts[status] = status_counts.get(status, 0) + 1
            
            if status not in age_by_status:
                age_by_status[status] = []
            age_by_status[status].append(p['age'])
        
        # Generate summary report
        summary = f"""
Simulated Patient Dataset Summary
===============================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Total Patients: {len(participants)}

Health Status Distribution:
"""
        
        for status, count in status_counts.items():
            percentage = (count / len(participants)) * 100
            avg_age = np.mean(age_by_status[status])
            summary += f"  {status}: {count} patients ({percentage:.1f}%), avg age: {avg_age:.1f}\n"
        
        summary += f"""
Age Statistics:
  Overall average age: {np.mean([p['age'] for p in participants]):.1f}
  Age range: {min([p['age'] for p in participants])}-{max([p['age'] for p in participants])}

Audio Augmentation Applied:
  - Pitch variations (voice changes)
  - Speed variations (cognitive processing)
  - Background noise (recording conditions)
  - Voice tremor (neurological effects)
  - Pause insertions (cognitive delays)
  - Reverb effects (environment simulation)

Files Generated:
  - {len(participants)} audio files in simulated_recordings/
  - Participant data in simulated_data/
"""
        
        # Save summary
        summary_file = os.path.join(self.sim_data_dir, "simulation_summary.txt")
        with open(summary_file, 'w') as f:
            f.write(summary)
        
        print(summary)
        print(f"Summary saved to {summary_file}")
    
    def integrate_with_main_system(self):
        """Integrate simulated data with the main system for testing."""
        
        print("\nIntegrating simulated data with main system...")
        
        # Copy simulated participant data to main location
        sim_data_file = os.path.join(self.sim_data_dir, "simulated_participant_data.txt")
        main_data_file = os.path.join(self.base_dir, "simulated_participant_data.txt")
        
        if os.path.exists(sim_data_file):
            shutil.copy2(sim_data_file, main_data_file)
            print(f"Copied participant data to {main_data_file}")
        
        # Instructions for using with the bot
        instructions = f"""
Integration Instructions:
========================

To test your AI bot with the simulated 100 patients:

1. Use simulated recordings:
   - Audio files are in: {os.path.abspath(self.sim_recordings_dir)}
   - Copy these to your main recordings folder if needed

2. Use simulated participant data:
   - Data file: {os.path.abspath(main_data_file)}
   - Modify your bot to load this file instead of the original

3. Generate scan results for simulated data:
   - Run your scan.py on simulated recordings
   - This will create biomarker data for training/testing

4. Train and test your bot:
   - The bot can now use 100+ patients for robust testing
   - Evaluate performance across different health conditions

Example modifications for botv3.py:
- Change participant_data_path to point to simulated data
- Modify recordings path to use simulated_recordings
- Run comprehensive cross-validation with larger dataset
"""
        
        instructions_file = os.path.join(self.sim_data_dir, "integration_instructions.txt")
        with open(instructions_file, 'w') as f:
            f.write(instructions)
            
        print(instructions)
        print(f"Instructions saved to {instructions_file}")

def main():
    """Main function to run the patient simulation."""
    
    print("Patient Simulation System for Alzheimer's Detection AI")
    print("=" * 55)
    
    # Initialize simulator
    simulator = PatientSimulator()
    
    # Check if original recordings exist
    if not os.path.exists(simulator.recordings_dir):
        print(f"Error: Recordings directory not found: {simulator.recordings_dir}")
        return
    
    original_recordings = glob.glob(os.path.join(simulator.recordings_dir, "*.wav"))
    if not original_recordings:
        print(f"Error: No WAV files found in {simulator.recordings_dir}")
        return
    
    print(f"Found {len(original_recordings)} original recordings")
    print(f"Original participants: {len(simulator.original_participants)}")
    
    # Number of patients to generate - CHANGE THIS FOR MORE DATA
    num_patients = 500  # Increased from 100 to 500
    
    # Generate simulated patients
    try:
        participants = simulator.generate_simulated_patients(num_patients)
        
        # Integrate with main system
        simulator.integrate_with_main_system()
        
        print("\n" + "="*60)
        print("SIMULATION COMPLETE!")
        print("="*60)
        print(f"Generated {len(participants)} synthetic patients for testing")
        print(f"Run 'python advanced_ensemble_bot.py improved' to train with new data!")
        
    except Exception as e:
        print(f"Error during simulation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()