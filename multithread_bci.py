import numpy as np
from scipy import signal
import pandas as pd
from matplotlib import style
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import matplotlib
import time
import sys
import brainflow
import serial
import serial.tools.list_ports

from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations
from brainflow.ml_model import MLModel, BrainFlowMetrics, BrainFlowClassifiers, BrainFlowModelParams
import threading

params = BrainFlowInputParams()
board_id = BoardIds.SYNTHETIC_BOARD.value
board = BoardShim(board_id, params)

eeg_channels = BoardShim.get_emg_channels(board_id)
sampling_rate = BoardShim.get_sampling_rate(board_id)
timestamp = BoardShim.get_timestamp_channel(board_id)

board.prepare_session()
board.start_stream()

n_samples = 1000

# Shared EEG data
eeg_data = np.zeros((len(eeg_channels), n_samples))

# Status of anger state
angry_bool = False

def get_data():
    while True:
        while board.get_board_data_count() < n_samples:
            time.sleep(0.005)
        data = board.get_current_board_data(n_samples)

        for count, channel in enumerate(eeg_channels):
            DataFilter.perform_bandstop(
                data[channel], sampling_rate, 48.0, 52.0, 4,
                FilterTypes.BUTTERWORTH.value, 0)  # bandstop 58 - 62

            eeg_data[count, :] = data[channel]

thread1 = threading.Thread(target=get_data, daemon=True)

# Start the threads
thread1.start()

for i in range(1000):
    time.sleep(1)
    print(eeg_data)