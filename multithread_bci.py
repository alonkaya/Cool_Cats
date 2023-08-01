import numpy as np
import time

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes
import threading

import tkinter as tk
from textblob import TextBlob

from gui import WhatsAppLikeApp
from face_recognition import VideoCamera, face_gen

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

camera = VideoCamera()

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

def start_gui():
    root = tk.Tk()
    app = WhatsAppLikeApp(root, camera)
    root.mainloop()

def start_face_recogition():
    face_gen(camera)


if __name__ == "__main__":
    thread1 = threading.Thread(target=get_data, daemon=True)
    thread2 = threading.Thread(target=start_gui, daemon=True)
    thread3 = threading.Thread(target=start_face_recogition, daemon=True)    

    thread1.start()
    thread2.start()
    thread3.start()

    thread1.join()
    thread2.join()
    thread3.join()