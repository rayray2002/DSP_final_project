import numpy as np
import time

from config import *


def get_input_device_index():
    for i in range(pyaudio.PyAudio().get_device_count()):
        if pyaudio.PyAudio().get_device_info_by_index(i)["maxInputChannels"] > 0:
            if (
                "Yeti Stereo Microphone"
                in pyaudio.PyAudio().get_device_info_by_index(i)["name"]
            ):
                DEVICE_INDEX = i
    print(pyaudio.PyAudio().get_device_info_by_index(DEVICE_INDEX)["name"])


def get_threshold(stream, CHUNK, dtype=FORMAT):
    print("getting ambient noise...")
    stream.start_stream()
    data = stream.read(CHUNK)
    stream.stop_stream()
    data = np.frombuffer(data, dtype=dtype)
    ambient_noise = np.average(np.abs(data))
    print("ambient noise:", ambient_noise)

    print("getting active noise...")
    time.sleep(1)
    stream.start_stream()
    data = stream.read(CHUNK)
    stream.stop_stream()
    data = np.frombuffer(data, dtype=dtype)
    active_noise = np.average(np.abs(data))
    print("active noise:", active_noise)

    threshold = ambient_noise * 0.7 + active_noise * 0.3
    print("threshold:", threshold)

    return threshold, ambient_noise, active_noise


def truncate_silence(data, threshold=100):
    start = 0
    for i in range(len(data)):
        if abs(data[i]) > threshold:
            start = i
            break

    end = len(data)
    for i in range(len(data) - 1, 0, -1):
        if abs(data[i]) > threshold:
            end = i
            break

    # print("start:", start, "end:", end)

    if start == 0 and end == len(data):
        return None, 0

    if end - start < 100:
        return None, 0

    data = data[start - 10 : end + 10]
    duration = len(data) / RATE

    return data, duration


def green(text):
    return f"\033[32m{text}\033[0m"


def red(text):
    return f"\033[31m{text}\033[0m"
