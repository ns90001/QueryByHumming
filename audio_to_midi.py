import numpy as np
import librosa
from midiutil import MIDIFile
from matplotlib import pyplot as plt
import mido
from mido import MidiFile, Message, MidiTrack
from query_by_humming import extract_vocals, mid2arry
import sounddevice as sd
import time
import crepe
import math

def preprocess_notes(notes, min_note_duration=4):
    """
    Merge short notes into previous long notes.
    """
    merged_notes = []
    prev_freq, prev_duration = None, None

    for freq, duration in notes:
        if prev_freq is not None and prev_duration is not None:
            if duration < min_note_duration:
                prev_duration += duration  # Merge duration to the previous note
            else:
                merged_notes.append([prev_freq, prev_duration])
                prev_freq, prev_duration = freq, duration
        else:
            prev_freq, prev_duration = freq, duration

    if prev_freq is not None and prev_duration is not None:
        merged_notes.append([prev_freq, prev_duration])

    return merged_notes

def round_to_nearest_semitone(freq):
    """
    Round the frequency to the nearest semitone.
    """
    # A4 reference frequency (440 Hz)
    A4_ref = 440.0

    # Calculate the ratio of the given frequency to A4 reference frequency
    ratio = freq / A4_ref

    # Calculate the number of semitones away from A4
    num_semitones = 12 * math.log2(ratio)

    # Round to the nearest integer number of semitones and calculate the corresponding frequency
    rounded_num_semitones = round(num_semitones)
    rounded_freq = A4_ref * (2 ** (rounded_num_semitones / 12))

    return rounded_freq

def generate_sine_wave(freq, duration, sampling_rate=44100):
    """
    Generate a sine wave of given frequency and duration.
    """
    t = np.linspace(0, duration, int(sampling_rate * duration), False)
    wave = np.sin(2 * np.pi * freq * t)
    return wave.astype(np.float32)


def play_notes(notes):
    """
    Play notes sequentially.
    """
    sampling_rate = 44100

    for note in notes:
        freq, duration = note
        duration = duration
        if duration > 0.08 and freq < 1000:
            wave = generate_sine_wave(round_to_nearest_semitone(freq), duration, sampling_rate)
            sd.play(wave, samplerate=sampling_rate)
            time.sleep(duration)


def audio_to_midi(audio_file, output_midi, window_size=2048, hop_size=512, volume_threshold=0, frequency_tolerance=10):
    # # Load audio file
    # y, sr = librosa.load(audio_file)

    # # Convert audio to spectrogram
    # spectrogram = librosa.stft(y, n_fft=window_size, hop_length=hop_size, window='hann')

    # # Convert magnitude spectrogram to decibels
    # spectrogram_db = librosa.amplitude_to_db(np.abs(spectrogram))

    # # Get frequencies corresponding to each bin
    # frequencies = librosa.fft_frequencies(sr=sr, n_fft=window_size)

    # # Filter out frequencies close to zero
    # valid_indices = frequencies > 0
    # frequencies = frequencies[valid_indices]
    # spectrogram_db = spectrogram_db[valid_indices, :]

    # Load the audio file
    y, sr = librosa.load(audio_file)

    # # Define parameters for STFT
    # window_size = 2048
    # hop_size = 512

    # Convert audio to spectrogram
    spectrogram = librosa.stft(y, n_fft=window_size, hop_length=hop_size, window='hann')

    # Convert magnitude spectrogram to decibels
    spectrogram_db = librosa.amplitude_to_db(np.abs(spectrogram))

    # Get frequencies corresponding to each bin
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=window_size)

    # Filter out frequencies close to zero
    valid_indices = frequencies > 0
    frequencies = frequencies[valid_indices]
    spectrogram_db = spectrogram_db[valid_indices, :]

    # # Apply a low-pass filter to remove overtones
    # cutoff_freq = 1000  # Adjust as needed
    # cutoff_bin = int(cutoff_freq * window_size / sr)
    # lower_cuttoff = 100
    # cutoff_bin_low = int(lower_cuttoff * window_size / sr)
    # spectrogram_db[cutoff_bin:, :] = np.min(spectrogram_db)  # Zero out higher frequencies
    # spectrogram_db[:cutoff_bin_low, :] = np.min(spectrogram_db)
    # # Optionally, convert back to magnitude spectrogram
    # # spectrogram_db = librosa.db_to_amplitude(spectrogram_db)

    # midi_duration = []
    # current_note = frequencies[0]
    # duration = 1

    # for i in range(spectrogram_db.shape[1]):
    #     max_index = np.argmax(spectrogram_db[:, i])
    #     freq = frequencies[max_index]
    #     volume_level = spectrogram_db[max_index, i]
    #     print(freq, volume_level)
    #     if volume_level < volume_threshold:
    #         # If volume is below threshold, consider it as silence
    #         current_note = None
    #         duration += 1
    #     elif current_note is not None and (freq / current_note > 1.06 or current_note / freq > 1.06):
    #         print(volume_level)
    #         midi_duration.append([current_note, duration])
    #         current_note = freq
    #         duration = 1
    #     elif current_note is None:
    #         current_note = freq
    #         duration = 1
    #     else:
    #         duration += 1

    # # Add the last note and its duration
    # if current_note is not None:
    #     midi_duration.append([current_note, duration])

    time, frequency, confidence, _ = crepe.predict(y, sr, viterbi=True, step_size=50, model_capacity="large")

    # Convert frequency to MIDI note number
    notes = []
    current_note = None
    note_start_time = 0
    for t, f in zip(time, frequency):
        if np.isnan(f):
            # Skip if frequency is NaN
            continue
        # f = round_to_nearest_semitone(f)
        print(f, current_note)
        if current_note is not None:
            print(f / current_note)
        if current_note is None or ((f / current_note) > 1.05 or (current_note / f) > 1.05):
            # Start a new note if frequency ratio is within range
            if current_note is not None:
                notes.append((current_note, t - note_start_time))
            current_note = f
            note_start_time = t

    # Append the last note
    if current_note is not None:
        notes.append((current_note, time[-1] - note_start_time))

    # Plot notes
    plt.subplot(2, 1, 1)
    librosa.display.specshow(spectrogram_db, sr=sr, hop_length=hop_size, x_axis='time', y_axis='log')
    plt.subplot(2, 1, 2)

    current_time = 0

    for note, duration in notes:
        plt.plot([current_time, current_time + duration], [note, note], color='blue', linewidth=3)
        current_time += duration

    plt.xlabel('Time')
    plt.ylabel('MIDI Note')
    plt.title('MIDI Duration Plot')
    plt.show()

    # # Create MIDI file
    # midi = MIDIFile(1)
    # track = 0
    # time = 0
    # tempo = 120  # BPM
    # midi.addTempo(track, time, tempo)

    # # Add notes to MIDI track
    # channel = 0
    # volume = 100
    # time = 0

    # for note in notes:
    #     if note is not None:
    #         duration = hop_size / sr  # Duration in seconds
    #         midi.addNote(track, channel, int(note), time, duration, volume)
    #     else:
    #         # Add silence
    #         duration = hop_size / sr  # Duration in seconds
    #         midi.addNote(track, channel, 0, time, duration, 0)
    #     time += duration  # Move time forward by note duration

    # # Write MIDI file
    # with open(output_midi, "wb") as midi_file:
    #     midi.writeFile(midi_file)

    # plt.figure(figsize=(10, 8))
    # plt.subplot(3, 1, 1)

    # a = mid2arry(midi)

    # librosa.display.specshow(spectrogram_db, sr=sr, hop_length=hop_size, x_axis='time', y_axis='log')
    # plt.subplot(3, 1, 2)
    # plt.plot(np.arange(len(notes)) * hop_size / sr, [note if note is not None else -1 for note in notes], marker='o', markersize=3, linestyle='-', color='b')
    # plt.subplot(3, 1, 3)
    # plt.plot(range(a.shape[0]), np.multiply(np.where(a>0, 1, 0), range(1, 89)), marker='.', markersize=1, linestyle='')
    # plt.show()

    return notes

# Example usage

def save_midi_duration(filename, midi_duration):
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    ticks_per_beat = mido.bpm2tempo(120) // 4  # Assuming 4 ticks per beat

    current_time = 0
    for note, duration in midi_duration:
        note = librosa.hz_to_midi(note)
        track.append(Message('note_on', note=int(note), velocity=64, time=current_time))
        track.append(Message('note_off', note=int(note), velocity=64, time=current_time + int(duration * ticks_per_beat)))
        current_time += int(duration * ticks_per_beat)

    mid.save(filename)

def expand_notes(notes):
    """
    Expand notes with integer durations into an expanded version.
    """
    expanded_notes = []
    for freq, duration in notes:
        expanded_notes.extend([freq] * duration)
    return expanded_notes
from librosa.display import specshow
audio_file = 'Let It Be - The Beatles [vocals] .wav'
# song_name = (audio_file.split('/')[-1]).split('.')[0]
# vfile = str(song_name) + " [vocals] .wav"
# # extract midi melody and add to output_dir directory
# # extract_vocals(audio_file, vfile)
# output_midi = 'CMajor.mid'
# # notes = audio_to_midi(audio_file, output_midi)
# print(f"MIDI file saved as {output_midi}")
# play_notes_sequence(notes)
# notes = [[440, 0.5], [523.25, 0.5], [659.25, 0.5]]  # A4, C5, E5
# save_midi_duration("output.mid", preprocess_notes(notes))
# play_notes(notes)

def extract_midi_notes_nn(audio_file, step_size=50, plot=False):

    y, sr = librosa.load(audio_file)
    time, frequencies, confidence, _ = crepe.predict(y, sr, viterbi=True, step_size=step_size, model_capacity="medium")

    notes = librosa.hz_to_midi(frequencies)

    if plot:   
        specshow(librosa.amplitude_to_db(np.abs(librosa.cqt(y, hop_length=1024, sr=sr))**2))
        notes = librosa.hz_to_midi(notes + np.finfo(float).eps).round()
        plt.step(np.arange(len(notes)), 
                notes - librosa.note_to_midi('C1'), 
                marker='|',
                label='Melody')
        plt.title('CQT spectrogram with melody overlay')
        plt.legend()
        plt.show()
    
    return notes

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
import librosa as lr
from librosa.display import specshow
from pysndfx import AudioEffectsChain

def pitch_hps(audio_samples,
              sample_rate=1000,
              window_length=4096,
              hop_length=1024,
              window=np.hanning,
              partials=5,
              plot=False):
    """Estimate the pitch contour in a monophonic audio signal."""

    f0s = []
    frequencies = np.fft.rfftfreq(window_length, 1 / sample_rate)
    window = window(window_length)
    pad = lambda a: np.pad(a, 
                           (0, window_length - len(a)),
                           mode='constant',
                           constant_values=0)
    
    # Low cut filter audio at 50 Hz.
    # audio_samples = AudioEffectsChain().highpass(50)(audio_samples)

    # Go through audio frame-by-frame.
    for i in range(0, len(audio_samples), hop_length):

        # Fourier transform audio frame.

        frame = window * pad(audio_samples[i:window_length + i])
        spectrum = np.fft.rfft(frame)

        # Downsample spectrum.
        spectra = []
        for n in range(1, partials + 1):
            s = sp.signal.resample(spectrum, len(spectrum) // n)
            spectra.append(s)

        # Truncate to most downsampled spectrum.
        l = min(len(s) for s in spectra)
        a = np.zeros((len(spectra), l), dtype=spectrum.dtype)
        for i, s in enumerate(spectra):
            a[i] += s[:l]

        # Multiply spectra per frequency bin.
        hps = np.product(np.abs(a), axis=0)

        # TODO Blur spectrum to remove noise and high-frequency content.
        #kernel = sp.signal.gaussian(9, 1)
        #hps = sp.signal.fftconvolve(hps, kernel, mode='same')

        # TODO Detect peaks with a continuous wavelet transform for polyphonic signals.
        #peaks = sp.signal.find_peaks_cwt(np.abs(hps), np.arange(1, 3))

        # Pick largest peak, it's likely f0.
        peak = np.argmax(hps)
        f0 = frequencies[peak]
        print(f0)
        f0s.append(f0)

        if plot:

            # Plot partial magnitudes individually.
            for s, ax in zip(spectra,
                             plt.subplots(len(spectra), sharex=True)[1]):
                ax.plot(np.abs(s))
            plt.suptitle('Partials')
            plt.show()

            # Plot combined spectra.
            plt.imshow(np.log(np.abs(a)), aspect='auto')
            plt.title('Spectra')
            plt.colorbar()
            plt.show()

            # Plot HPS peak.
            plt.plot(np.arange(len(hps)), np.abs(hps))
            plt.scatter(peak, np.abs(hps[peak]), color='r')
            plt.title('HPS peak')
            plt.show()
            return

    f0s = np.array(f0s)

    # Median filter out noise.
    f0s = sp.signal.medfilt(f0s, [21])

    return f0s

# audio_file = 'song_database/New Recording 17.mp3'
# y, sr = librosa.load(audio_file)
# f0s = pitch_hps(y, sample_rate=sr, plot=True)
# print(f0s)
# specshow(lr.amplitude_to_db(np.abs(lr.cqt(y, hop_length=1024, sr=sr))**2))
# notes = lr.hz_to_midi(f0s + np.finfo(float).eps).round()
# plt.step(np.arange(len(f0s)), 
#          notes - lr.note_to_midi('C1'), 
#          marker='|',
#          label='Melody')
# plt.title('CQT spectrogram with melody overlay')
# plt.legend();
