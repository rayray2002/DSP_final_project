import os
import sys

import librosa
import numpy as np
import threading

from config import *
from utils import *

np.set_printoptions(threshold=sys.maxsize, linewidth=np.inf)


class DTWDetector:
    def __init__(self):
        self.keyword_MFCCs = []
        self.dist_threshold = 500
        self.length_threshold = 30

        for filename in os.listdir("keyword"):
            if filename.endswith(".wav"):
                filepath = os.path.join("keyword", filename)
                MFCC = self.get_MFCC(filepath)
                self.keyword_MFCCs.append(MFCC)
                # print(MFCC.shape)

        # self.calibrate()

    def get_MFCC(self, filepath):
        """
        get MFCC features from audio file
        """
        signal, sr = librosa.load(filepath, sr=RATE)
        signal = librosa.util.normalize(signal)
        return librosa.feature.mfcc(
            y=signal, sr=RATE, n_mfcc=NUM_FILTERS, hop_length=HOP_LENGTH, n_fft=N_FFT
        ).T

    def dtw(self, x, y, dist):
        """
        Subsequence Dynamic Time Warping
        :param x: subsequence to find
        :param y: input sequence
        :param dist: Distance function
        :return: Distance, accumulated cost matrix
        """
        assert len(x)
        assert len(y)
        assert dist

        C = np.zeros((len(x), len(y)))
        for i in range(len(x)):
            for j in range(len(y)):
                C[i, j] = dist(x[i], y[j])

        D = np.zeros((len(x), len(y)))

        for i in range(len(x)):
            D[i, 0] = np.sum(C[: i + 1, 0])

        for j in range(len(y)):
            D[0, j] = C[0, j]

        for i in range(1, len(x)):
            for j in range(1, len(y)):
                D[i, j] = C[i, j] + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])

        end = np.argmin(D[-1, :])

        # backtrack
        distances = D[-1, end]
        i = len(x) - 1
        j = end
        k = 0

        while i > 0:
            k += 1

            if j == 0:
                i = 0
                break

            if D[i, j] == D[i - 1, j - 1] + C[i, j]:
                i -= 1
                j -= 1
            elif D[i, j] == D[i - 1, j] + C[i, j]:
                i -= 1
            else:
                j -= 1

        start = j
        distances /= k

        return distances, D, start, end

    def detect_keyword(self, fbank):
        """
        Detect keyword in audio data
        :param fbank: Log Mel filterbank energies
        :return: Prediction, distance
        """

        results = [None] * len(self.keyword_MFCCs)

        def dtw_thread(keyword_MFCC, fbank, index):
            out = self.dtw(
                keyword_MFCC, fbank, dist=lambda x, y: np.linalg.norm(x - y, ord=1)
            )
            results[index] = out

        threads = []
        for i, keyword_MFCC in enumerate(self.keyword_MFCCs):
            t = threading.Thread(target=dtw_thread, args=(keyword_MFCC, fbank, i))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        best = min(results, key=lambda x: x[0])
        (dist, D, start, end) = best

        min_dist = dist

        pred = end - start > self.length_threshold and min_dist < self.dist_threshold

        return pred, best

    def calibrate(self):
        """
        Calibrate the distance threshold
        """

        dists = []
        for i, mfcc1 in enumerate(self.keyword_MFCCs):
            # print(i, mfcc1.shape)
            for j, mfcc2 in enumerate(self.keyword_MFCCs):
                dist, D, start, end = self.dtw(
                    mfcc1, mfcc2, dist=lambda x, y: np.linalg.norm(x - y, ord=1)
                )
                dists.append(dist)

                # print(f"{i:2d} {j:2d} {dist:.2e} {start:2d} {end:2d}")
                # print(np.array2string(D, precision=2, separator=", "))

        dist_threshold = np.mean(dists) * 1.5
        print(f"Distance threshold: {dist_threshold:.2e}")

        self.dist_threshold = dist_threshold


if __name__ == "__main__":
    detector = DTWDetector()

    import time

    import pyaudio

    RECORD_SECONDS = 1
    CHUNK = int(RATE * RECORD_SECONDS)
    get_input_device_index()

    p = pyaudio.PyAudio()
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
        input_device_index=DEVICE_INDEX,
    )

    input_threshold = INPUT_THRESHOLD
    # input_threshold = get_threshold(stream, CHUNK, np.float32)
    # time.sleep(1)

    while True:
        print("Listening...")
        stream.start_stream()
        data = stream.read(CHUNK)
        stream.stop_stream()

        data = np.frombuffer(data, dtype=np.float32)
        data, duration = truncate_silence(data, input_threshold)

        # if data is None or duration < 0.1:
        #     print("No audio input")
        #     continue

        fbank = librosa.feature.mfcc(
            y=data, sr=RATE, n_mfcc=NUM_FILTERS, hop_length=HOP_LENGTH, n_fft=N_FFT
        ).T
        pred, (dist, D, start, end) = detector.detect_keyword(fbank)

        print(data.shape, fbank.shape)
        # print(np.array2string(D, precision=0, separator=" "))

        if pred:
            print(green(f"Dist: {dist:.2e} {start:2d} {end:2d}"))
        else:
            print(red(f"Dist: {dist:.2e} {start:2d} {end:2d}"))

        time.sleep(1)
