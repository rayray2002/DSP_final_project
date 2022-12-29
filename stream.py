from queue import Queue

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pyaudio

from config import *
from dtw import DTWDetector
from utils import *


class StreamPrediction:
    def __init__(self, detector):
        # Recording parameters
        self.sr = RATE
        self.chunk_duration = 0.5
        self.chunk_samples = int(self.sr * self.chunk_duration)
        self.window_duration = 1
        self.window_samples = int(self.sr * self.window_duration)
        self.silence_threshold = 0.002

        # Data structures and buffers
        self.queue = Queue()
        self.data = np.zeros(self.window_samples, dtype=np.float32)

        # Plotting parameters
        self.change_bkg_frames = 2
        self.change_bkg_counter = 0
        self.change_bkg = False
        self.max_amp = 0.1

        # detector
        self.starburst_detector = detector("starburst")
        self.star_detector = detector("star")

    def start_stream(self):
        """
        Start audio data streaming from microphone
        :return: None
        """

        stream = pyaudio.PyAudio().open(
            format=FORMAT,
            channels=CHANNELS,
            rate=self.sr,
            input=True,
            frames_per_buffer=self.chunk_samples,
            input_device_index=DEVICE_INDEX,
            stream_callback=self.callback,
        )

        stream.start_stream()
        print("Listening...")

        try:
            # fbank = np.zeros((1, NUM_FILTERS))
            while True:
                data = self.queue.get()
                rms = np.sqrt(np.mean(np.square(data)))
                # data_T, duration = truncate_silence(data, self.silence_threshold)

                # print(data.shape, np.mean(np.abs(data)))

                if rms > self.silence_threshold:
                    # normalize
                    data_norm = librosa.util.normalize(data)
                    mfcc = get_mfcc(data_norm)  # print(fbank.shape)
                    starburst_pred, (starburst_dist, _, starburst_start, starburst_end) = self.starburst_detector.detect_keyword(mfcc)
                    star_pred, (star_dist, _, star_start, star_end) = self.star_detector.detect_keyword(mfcc)
                    
                else:
                    starburst_pred, (starburst_dist, _, starburst_start, starburst_end) = False, (0, 0, 0, 0)
                    star_pred, (star_dist, _, star_start, star_end) = False, (0, 0, 0, 0)
                # print(starburst_pred, dist, starburst_start, starburst_end)
                
                pred = starburst_pred
                if star_pred and star_dist < starburst_dist:
                    pred = False
                    
                self.plotter(data, pred, starburst_dist)
                
                if starburst_dist > 0:
                    if starburst_dist < self.starburst_detector.dist_threshold:
                        dist_text = green(f"dist: {starburst_dist:7.1f}")
                    else:
                        dist_text = red(f"dist: {starburst_dist:7.1f}")

                    length = starburst_end - starburst_start
                    if length > self.starburst_detector.length_threshold:
                        length_text = green(f"length: {length:2d}")
                    else:
                        length_text = red(f"length: {length:2d}")

                    q_size = self.queue.qsize()
                    if q_size > 0:
                        Q_text = red(f"Q: {q_size:2d}")
                    else:
                        Q_text = green(f"Q: {q_size:2d}")

                    text = f"{dist_text}, {length_text}, {Q_text}, "
                    
                    star_text = f"star: dist: {star_dist:7.1f}, length: {(star_end - star_start):2d}"
                    if star_pred:
                        text += green(star_text)
                    else:
                        text += red(star_text)

                    if pred:
                        text += bold(blue("  Keyword detected!"))
                        
                    print(text)
                else:
                    print(".")

        except (KeyboardInterrupt, SystemExit):
            stream.stop_stream()
            stream.close()

    def callback(self, in_data, frame_count, time_info, status):
        """
        Obtain the data from buffer and load it to queue
        :param in_data: Daa buffer
        :param frame_count: Frame count
        :param time_info: Time information
        :param status: Status
        :return:
        """

        data0 = np.frombuffer(in_data, dtype=np.float32)
        rms = np.sqrt(np.mean(np.square(data0)))
        # print(rms)

        # if rms < self.silence_threshold:
        #     print(".", sep="", end="", flush=True)
        # else:
        #     print("-", sep="", end="", flush=True)

        self.data = np.append(self.data, data0)

        while len(self.data) > self.window_samples:
            self.data = self.data[-self.window_samples :]
            self.queue.put(self.data)

        # print(self.queue.qsize(), end="")

        return in_data, pyaudio.paContinue

    def plotter(self, data, pred, score):
        """
        Plot waveform, filterbank energies and keyword presence
        :param data: Audio data array
        :param fbank: Log Mel filterbank energies
        :param pred: Prediction
        :return:
        """

        plt.clf()

        # Wave
        self.max_amp = max(self.max_amp, np.max(np.abs(data)))
        plt.subplot(211)
        plt.plot(data[-len(data) :])
        plt.text(
            x=RATE / 2,
            y=0.95 * self.max_amp,
            s=f"rms: {np.sqrt(np.mean(np.square(data))):.4f}",
            horizontalalignment="center",
            verticalalignment="top",
        )
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.ylim(-self.max_amp, self.max_amp)
        plt.ylabel("Amplitude")

        # print(fbank.shape)
        # MFCC
        # plt.subplot(312)
        # plt.imshow(mfcc[-mfcc.shape[0] :, :].T, aspect="auto")
        # plt.gca().xaxis.set_major_locator(plt.NullLocator())
        # plt.gca().invert_yaxis()
        # plt.ylim(0, NUM_FILTERS * 3)
        # plt.ylabel("MFCC")

        # keyword detection
        plt.subplot(212)
        ax = plt.gca()
        ax.yaxis.set_major_locator(plt.NullLocator())
        plt.ylabel(f"Q: {self.queue.qsize()}")

        ax.text(
            x=0.5,
            y=0.95,
            s=f"score: {score:.1f}",
            horizontalalignment="center",
            verticalalignment="top",
        )

        if pred == 1:
            self.change_bkg = True

        if self.change_bkg and self.change_bkg_counter < self.change_bkg_frames:
            ax.set_facecolor("lightgreen")

            ax.text(
                x=0.5,
                y=0.5,
                s="DETECT!",
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=30,
                color="red",
                fontweight="bold",
                transform=ax.transAxes,
            )

            self.change_bkg_counter += 1
        else:
            ax.set_facecolor("salmon")
            self.change_bkg = False
            self.change_bkg_counter = 0

        plt.tight_layout()
        plt.pause(0.001)


if __name__ == "__main__":
    get_input_device_index()
    audio_stream = StreamPrediction(DTWDetector)
    audio_stream.start_stream()
