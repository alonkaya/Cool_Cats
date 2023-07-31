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

fig, axes = plt.subplots(2, 1, figsize=(7, 7))
axes[0].set_ylabel("Voltage (mV)", fontsize=10)
axes[0].set_xlabel("Time (sec)", fontsize=10)

ports = serial.tools.list_ports.comports()
print([port.device for port in ports])

BoardShim.enable_dev_board_logger()

# optional. take this out for initial setup for your board.
BoardShim.disable_board_logger()

# use synthetic board for demo
params = BrainFlowInputParams()
params.serial_port = "/dev/ttyACM0"

# board_id = BoardIds.GANGLION_BOARD
# board = BoardShim(board_id, params)

board_id = BoardIds.SYNTHETIC_BOARD.value
board = BoardShim(board_id, params)


eeg_channels = BoardShim.get_emg_channels(board_id)
print(eeg_channels)
sampling_rate = BoardShim.get_sampling_rate(board_id)
timestamp = BoardShim.get_timestamp_channel(board_id)

board.prepare_session()
board.start_stream()

def main(i):

    keep_alive = True

    eeg1, eeg2, eeg3, eeg4 = list(), list(), list(), list()
    timex = list()  # list to store timestamp

    while keep_alive == True:
        time.sleep(1)

        # ensures that all data shape is the same
        # while board.get_board_data_count() < 250:
        #     time.sleep(0.005)
        data = board.get_current_board_data(250)

        # creating a dataframe of the eeg data to extract eeg values later
        # eegdf = pd.DataFrame(np.transpose(data[eeg_channels]))
        # eegdf_col_names = [f'Ch{idx}' for idx in range(1,5)]
        # eegdf.columns = eegdf_col_names

        # making another dataframe for the timestamps to access later
        # timedf = pd.DataFrame(np.transpose(data[timestamp]))

        for count, channel in enumerate(eeg_channels):
            DataFilter.perform_bandstop(
                data[channel], sampling_rate, 48.0, 52.0, 4,
                FilterTypes.BUTTERWORTH.value, 0)  # bandstop 58 - 62
            # DataFilter.perform_bandpass(
            #     data[channel], sampling_rate, 11.0, 31.0, 4,
            #     FilterTypes.BESSEL.value, 0)

        f1, Pxx_den1 = signal.welch(data[eeg_channels[0]], sampling_rate)
        alpha_power = np.sum(Pxx_den1[np.where((f1 >= 8.0) & (f1 <= 12.0))])
        # f2, Pxx_den2 = signal.welch(data[eeg_channels[0]], sampling_rate)

        # appending eeg data to lists
        eeg1.extend(data[eeg_channels[0]])
        eeg2.extend(data[eeg_channels[1]])
        eeg3.extend(data[eeg_channels[2]])
        eeg4.extend(data[eeg_channels[3]])

        timex.extend(data[timestamp])  # timestamps

        axes[0].clear()
        axes[1].clear()

        # plotting eeg data
        axes[0].plot(timex, eeg1, label="Ch1", color='C0')
        # axes[0].set_ylim((-500, 500))
        # axes[0].plot(timex, eeg2, label="Ch2", color='C1')
        # axes[0].plot(timex, eeg3, label="Ch3", color='C2')
        # axes[0].plot(timex, eeg4, label="Ch4", color='C3')


        axes[1].plot(f1, Pxx_den1, 'C0')
        axes[1].set_title(alpha_power)


        # axes[1].plot(f2, Pxx_den2, 'C1')
        axes[1].set_yscale('log')
        axes[1].set_xlabel('log freq')


        keep_alive = False  # resetting stream so that matplotlib can plot data

        anger_value = 0
        if alpha_power > 2000:
            axes[0].set_title('Angry :(')
        else:
            axes[0].set_title('Happy :)')


    # board.stop_stream()
    # board.release_session()


# Continuously calls function until keyboard interrupt
ani = FuncAnimation(fig, main, interval=1)
plt.tight_layout()
plt.autoscale(enable=True, axis="y", tight=True)
plt.show()
