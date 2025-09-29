# =============================
# Music Generation with AI - Complete Local Version
# =============================

import glob
import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import Counter
import os
import urllib.request
import zipfile
import requests
import subprocess

from music21 import converter, instrument, note, chord, stream
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

# =============================
# Setup & Data Collection
# =============================
UPLOAD_DIR = './midi_dataset'
os.makedirs(UPLOAD_DIR, exist_ok=True)

def create_synthetic_training_data():
    """Create synthetic training data programmatically"""
    print("🎹 Creating synthetic training data...")
    
    # Create multiple simple melodies
    melodies = [
        # Simple C major scale
        ['C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4', 'C5'],
        # Simple chords
        ['C4', 'E4', 'G4', 'C5', 'G4', 'E4', 'C4'],
        # Arpeggio
        ['C4', 'E4', 'G4', 'C5', 'G4', 'E4', 'C4'],
        # Another simple pattern
        ['G4', 'A4', 'B4', 'C5', 'B4', 'A4', 'G4']
    ]
    
    # Create MIDI files from melodies
    for i, melody in enumerate(melodies):
        s = stream.Stream()
        for pitch in melody:
            n = note.Note(pitch)
            n.duration.quarterLength = 1.0
            s.append(n)
        s.write('midi', fp=os.path.join(UPLOAD_DIR, f'synthetic_{i}.mid'))
    
    # Return the combined notes for training
    all_notes = []
    for melody in melodies:
        all_notes.extend(melody)
    all_notes.extend(all_notes * 3)  # Repeat to get more data
    
    print(f"✅ Created {len(melodies)} synthetic MIDI files")
    print(f"📊 Total training notes: {len(all_notes)}")
    return all_notes

def get_enhanced_notes(data_dir, max_files=50):
    """Enhanced MIDI parsing with comprehensive fallbacks"""
    notes = []
    files_list = glob.glob(os.path.join(data_dir, '**', '*.mid'), recursive=True) + \
                glob.glob(os.path.join(data_dir, '**', '*.midi'), recursive=True)
    
    print(f"🔍 Found {len(files_list)} MIDI files")
    
    if len(files_list) == 0:
        print("⚠️ No MIDI files found. Using synthetic data...")
        return create_synthetic_training_data()
    
    files_list = files_list[:max_files]
    successful_files = 0

    for file in tqdm(files_list, desc="Parsing MIDI"):
        try:
            midi = converter.parse(file)
            notes_to_parse = midi.flat.notes
            
            file_notes = []
            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    file_notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    file_notes.append('.'.join(str(n) for n in element.normalOrder))
            
            if len(file_notes) > 10:  # Only use files with sufficient notes
                notes.extend(file_notes)
                successful_files += 1
                
        except Exception as e:
            print(f'❌ Error parsing {os.path.basename(file)}: {e}')
            continue
    
    print(f"✅ Successfully parsed {successful_files}/{len(files_list)} files")
    
    if len(notes) < 100:  # If we don't have enough notes, use synthetic data
        print("⚠️ Insufficient notes extracted. Augmenting with synthetic data...")
        synthetic_notes = create_synthetic_training_data()
        notes.extend(synthetic_notes)
    
    print(f"📊 Total notes for training: {len(notes)}")
    print(f"🎹 Unique tokens: {len(set(notes))}")
    
    # Show most common notes
    common_notes = Counter(notes).most_common(10)
    print("🏆 Most common notes:", common_notes)
    
    return notes

# =============================
# Parameters
# =============================
SEQUENCE_LENGTH = 30
BATCH_SIZE = 32
EPOCHS = 15
MODEL_SAVE_PATH = './lstm_music_model.h5'
GENERATED_MIDI = './generated_music.mid'
DATA_DIR = UPLOAD_DIR

# =============================
# Sequence Preparation
# =============================
def prepare_sequences(notes, sequence_length=SEQUENCE_LENGTH):
    if len(notes) < sequence_length + 1:
        print(f"⚠️ Warning: Only {len(notes)} notes available, need at least {sequence_length + 1}")
        # Duplicate notes to get minimum required
        notes = notes * (sequence_length // len(notes) + 2)
    
    pitchnames = sorted(set(notes))
    n_vocab = len(pitchnames)
    note_to_int = {note: number for number, note in enumerate(pitchnames)}
    int_to_note = {number: note for number, note in enumerate(pitchnames)}

    network_input = []
    network_output = []

    for i in range(0, len(notes) - sequence_length):
        seq_in = notes[i:i + sequence_length]
        seq_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in seq_in])
        network_output.append(note_to_int[seq_out])

    n_patterns = len(network_input)
    print(f'📈 Created {n_patterns} training sequences.')
    print(f'🎯 Vocabulary size: {n_vocab}')

    if n_patterns == 0:
        raise ValueError("No training sequences could be created! Need more data.")

    network_input = np.reshape(network_input, (n_patterns, sequence_length))
    network_output = to_categorical(network_output, num_classes=n_vocab)

    return network_input, network_output, note_to_int, int_to_note, n_vocab

# =============================
# Model Architecture
# =============================
def create_improved_network(sequence_length, n_vocab):
    """Simplified LSTM model for smaller datasets"""
    model = Sequential([
        tf.keras.layers.Input(shape=(sequence_length, 1)),
        LSTM(128, return_sequences=True),
        Dropout(0.2),
        LSTM(64),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(n_vocab, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', 
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  metrics=['accuracy'])
    return model

# =============================
# Training with Callbacks
# =============================
def train_with_callbacks(model, network_input, network_output, n_vocab,
                        model_save_path=MODEL_SAVE_PATH, 
                        epochs=EPOCHS, batch_size=BATCH_SIZE):
    X = np.reshape(network_input, (network_input.shape[0], network_input.shape[1], 1))
    X = X / float(n_vocab)
    
    # Adjust validation split based on available data
    n_samples = len(network_input)
    if n_samples < 100:
        validation_split = 0.0  # No validation for very small datasets
        print("⚠️ Small dataset: Disabling validation split")
    else:
        validation_split = 0.1
    
    callbacks = [
        ModelCheckpoint(model_save_path, monitor='loss', verbose=1, 
                       save_best_only=True, mode='min'),
        EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    ]
    
    print(f"🚀 Starting training with {n_samples} samples...")
    history = model.fit(X, network_output, 
                       epochs=epochs, 
                       batch_size=min(batch_size, n_samples),
                       callbacks=callbacks,
                       validation_split=validation_split,
                       verbose=1)
    
    return history

# =============================
# Generation with Temperature
# =============================
def generate_notes_with_temperature(model, network_input, int_to_note, note_to_int, 
                                  n_vocab, sequence_length=SEQUENCE_LENGTH, 
                                  generate_length=100, temperature=1.0):
    """Generate notes with temperature sampling"""
    if len(network_input) == 0:
        # Start with a random pattern if no network input
        available_notes = list(int_to_note.values())
        start_pattern = [note_to_int[note] for note in available_notes[:sequence_length]]
    else:
        start = np.random.randint(0, len(network_input)-1)
        start_pattern = list(network_input[start])
    
    pattern_norm = np.array(start_pattern) / float(n_vocab)
    
    prediction_output = []
    for note_index in range(generate_length):
        x = np.reshape(pattern_norm, (1, len(pattern_norm), 1))
        prediction = model.predict(x, verbose=0)[0]
        
        # Apply temperature
        prediction = np.log(prediction + 1e-8) / temperature
        exp_preds = np.exp(prediction)
        prediction = exp_preds / np.sum(exp_preds)
        
        # Sample from distribution
        index = np.random.choice(range(n_vocab), p=prediction)
        result = int_to_note[index]
        prediction_output.append(result)
        
        pattern_norm = np.append(pattern_norm[1:], index/float(n_vocab))
    
    return prediction_output

# =============================
# MIDI Creation
# =============================
def create_midi(prediction_output, output_file=GENERATED_MIDI):
    offset = 0
    output_notes = []
    
    for pattern in prediction_output:
        if '.' in pattern:  # Chord
            notes_in_chord = pattern.split('.')
            notes_obj = []
            for current_note in notes_in_chord:
                try:
                    new_note = note.Note(int(current_note))
                except:
                    try:
                        new_note = note.Note(current_note)
                    except:
                        new_note = note.Note('C4')  # Fallback note
                notes_obj.append(new_note)
            new_chord = chord.Chord(notes_obj)
            new_chord.offset = offset
            output_notes.append(new_chord)
        else:  # Single note
            try:
                new_note = note.Note(pattern)
            except:
                new_note = note.Note('C4')  # Fallback note
            new_note.offset = offset
            output_notes.append(new_note)
        offset += 0.5  # Move to next time position
    
    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp=output_file)
    print(f'💾 Generated MIDI: {output_file}')

# =============================
# Main Execution
# =============================
if __name__ == '__main__':
    print("🎵 MUSIC GENERATION WITH AI - STARTING 🎵")
    
    try:
        # 1. Load and preprocess data
        print("\n1. 📥 Loading MIDI data...")
        notes = get_enhanced_notes(DATA_DIR)
        
        # 2. Prepare sequences
        print("\n2. 🔄 Preparing sequences...")
        network_input, network_output, note_to_int, int_to_note, n_vocab = prepare_sequences(notes)
        
        # 3. Build model
        print("\n3. 🏗️ Building model...")
        model = create_improved_network(SEQUENCE_LENGTH, n_vocab)
        model.summary()
        
        # 4. Train model
        print("\n4. 🚀 Training model...")
        history = train_with_callbacks(model, network_input, network_output, n_vocab)
        
        # 5. Generate music
        print("\n5. 🎹 Generating music...")
        generated = generate_notes_with_temperature(
            model, network_input, int_to_note, note_to_int, n_vocab, 
            generate_length=50, temperature=0.8
        )
        create_midi(generated, GENERATED_MIDI)
        
        print("\n" + "="*60)
        print("🎵 MUSIC GENERATION COMPLETE! 🎵")
        print("="*60)
        print(f"💾 Generated MIDI: {GENERATED_MIDI}")
        print("🎹 Open the MIDI file with any music player to listen!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("💡 Trying to create a simple fallback melody...")
        
        # Create a simple fallback melody
        simple_melody = ['C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4', 'C5'] * 5
        create_midi(simple_melody, GENERATED_MIDI)
        print(f"✅ Created fallback melody: {GENERATED_MIDI}")