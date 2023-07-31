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

import tkinter as tk
from tkinter import messagebox
from textblob import TextBlob

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

# Status of anger state
camera = VideoCamera()


class WhatsAppLikeApp:
    def __init__(self, root, polarity_threshold=-0.3):
        self.root = root
        self.root.title("WhatsApp-like GUI")

        self.title_frame = tk.Frame(root)
        self.title_frame.pack(side=tk.TOP, fill=tk.X)
        
        # Add your picture here and provide the correct path
        self.photo = tk.PhotoImage(file='chat_image.png')
        self.photo = self.photo.subsample(10, 10)  # Resize the picture (change the subsample values to resize differently)
        self.photo_label = tk.Label(self.title_frame, image=self.photo)
        self.photo_label.pack(side=tk.LEFT, padx=5, pady=5)

        self.title_label = tk.Label(self.title_frame, text="Girlfriend <3", font=("Helvetica", 14, "bold"))
        self.title_label.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.message_frame = tk.Frame(root)
        self.message_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        self.typing_area = tk.Text(root, height=3, wrap=tk.WORD)
        self.typing_area.pack(side=tk.BOTTOM, fill=tk.BOTH, padx=5, pady=5)
        
        self.send_button = tk.Button(root, text="Send", command=self.send_message)
        self.send_button.pack(side=tk.BOTTOM, pady=5)

        self.message_list = []
        self.current_side = "right"
        self.current_row = 0

        self.polarity_threshold = polarity_threshold
        
    def send_message(self):
        message = self.typing_area.get("1.0", tk.END).strip()
        print(camera.pred)
        if message:
            if self.is_angry(message):
                self.show_custom_popup(message)
            else:
                self.typing_area.delete("1.0", tk.END)

                side = self.current_side
                self.current_side = "left" if self.current_side == "right" else "right"
                
                message_label = tk.Label(self.message_frame, text=message, bg="lightgreen", padx=10, pady=5, wraplength=200)
                
                # If it's on the right side, stick it to the right border of the window and anchor it to the east
                if side == "right":
                    message_label.grid(row=self.current_row, column=1, padx=(0, 10), pady=5, sticky="e")
                    self.message_frame.grid_columnconfigure(1, weight=1)  # Make the second column expand to the right
                else:
                    message_label.grid(row=self.current_row, column=0, padx=(10, 0), pady=5, sticky="w")

                self.current_row += 1
                
                self.message_list.append(message_label)

    def show_custom_popup(self, message):
        popup_window = tk.Toplevel(self.root)
        popup_window.title("Angry Alert")

        label = tk.Label(popup_window, text="Hey there big fellow, I've noticed you're a bit angry. Would you like to take a minute to calm and rephrase after you've cooled?")
        label.pack(padx=10, pady=10)

        send_anyway_button = tk.Button(popup_window, text="Send Anyway", command=lambda: self.send_message_after_popup(message, popup_window))
        send_anyway_button.pack(side=tk.LEFT, padx=5, pady=5)

        try_again_button = tk.Button(popup_window, text="Try Again Later", command=popup_window.destroy)
        try_again_button.pack(side=tk.LEFT, padx=5, pady=5)

    def send_message_after_popup(self, message, popup_window):
        popup_window.destroy()  # Close the pop-up window
        self.typing_area.delete("1.0", tk.END)

        side = self.current_side
        self.current_side = "left" if self.current_side == "right" else "right"
        
        message_label = tk.Label(self.message_frame, text=message, bg="lightgreen", padx=10, pady=5, wraplength=200)
        
        # If it's on the right side, stick it to the right border of the window and anchor it to the east
        if side == "right":
            message_label.grid(row=self.current_row, column=1, padx=(0, 10), pady=5, sticky="e")
            self.message_frame.grid_columnconfigure(1, weight=1)  # Make the second column expand to the right
        else:
            message_label.grid(row=self.current_row, column=0, padx=(10, 0), pady=5, sticky="w")

        self.current_row += 1
        
        self.message_list.append(message_label)

    def is_angry(self, message):
        blob_text = TextBlob(message)
        sentiment = blob_text.sentiment
        polarity = sentiment.polarity
        print(polarity)
        if polarity < self.polarity_threshold and ('Angry' in camera.pred_buffer or
                                                   'Sad' in camera.pred_buffer or
                                                   'Disgust' in camera.pred_buffer):
            return True
        else:
            return False

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
    app = WhatsAppLikeApp(root)
    root.mainloop()

def start_face_recogition():
    face_gen(camera)

thread1 = threading.Thread(target=get_data, daemon=True)
thread2 = threading.Thread(target=start_gui, daemon=True)
thread3 = threading.Thread(target=start_face_recogition, daemon=True)

# Start the threads
thread1.start()
thread2.start()
thread3.start()

if __name__ == "__main__":
    for i in range(1000):
        time.sleep(1)
        # print(eeg_data)