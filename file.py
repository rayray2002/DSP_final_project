import librosa
import sys

from config import *
from utils import *
from dtw import DTWDetector

filename = sys.argv[1]
signal, sr = librosa.load(filename, sr=RATE)
# signal = signal / np.max(np.abs(signal))
signal = librosa.util.normalize(signal)
mfcc = get_mfcc(signal)

detector = DTWDetector()
pred, (dist, D, start, end) = detector.detect_keyword(mfcc)

start = start * HOP_LENGTH / RATE
end = end * HOP_LENGTH / RATE

if pred:
    print(green(f"Detected, dist: {dist:.1f}, start: {start:.3f}, end: {end:.3f}"))
else:
    print(red(f"Not detected, dist: {dist:.1f}, start: {start:.3f}, end: {end:.3f}"))
