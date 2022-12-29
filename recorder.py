# record audio files into /keyword/ folder

import time
import wave
import soundfile as sf

import librosa
import numpy as np
import pyaudio
import sys

from config import *
from utils import *

get_input_device_index()
RECORD_SECONDS = 1
CHUNK = int(RATE * RECORD_SECONDS)

audio = pyaudio.PyAudio()

prefix = time.strftime("%m%d-%H%M")
count = 0

if len(sys.argv) > 1:
    FOLDER = sys.argv[1]
else:
    FOLDER = "keyword"


def write_audio(data):
    global count
    count += 1
    filename = f"{FOLDER}/{prefix}-{count}.wav"
    wf = wave.open(filename, "wb")
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
    wf.setframerate(RATE)
    wf.writeframes(data)
    wf.close()


# start Recording
stream = audio.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    frames_per_buffer=CHUNK,
    input_device_index=DEVICE_INDEX,
)

# get ambient noise
threshold, _, _ = get_threshold(stream, CHUNK, dtype=np.float32)
threshold = 0.01

while True:
    print("Recording...")
    stream.start_stream()
    data = stream.read(CHUNK)
    stream.stop_stream()

    data = np.frombuffer(data, dtype=np.float32)
    # data = librosa.effects.pitch_shift(data, sr=RATE, n_steps=4)

    rms = np.sqrt(np.mean(data**2))
    print("rms:", rms)

    data, duration = truncate_silence(data, threshold)
    # duration = librosa.get_duration(data, RATE)

    if data is None or duration < 0.1:
        continue

    print("duration:", duration)
    
    # data = data / np.max(np.abs(data))
    data = librosa.util.normalize(data)
    
    count += 1
    sf.write(f"{FOLDER}/{prefix}-{count}.wav", data, RATE)

    # librosa.output.write_wav(f"{FOLDER}/{prefix}-{count}.wav", data, RATE)
    
    # data = data / np.max(np.abs(data))x
    
    # data = data.astype(np.int16).tobytes()
    # write_audio(data)

    print()
    time.sleep(1)
