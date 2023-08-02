import tkinter as tk
from textblob import TextBlob

angry_text_threshold = -0.19

class WhatsAppLikeApp:
    def __init__(self, root, camera):
        self.root = root
        self.root.title("WhatsApp-like GUI")

        self.title_frame = tk.Frame(root)
        self.title_frame.pack(side=tk.TOP, fill=tk.X)
        
        # Add your picture here and provide the correct path
        self.photo = tk.PhotoImage(file='Gal_Gadot.png')
        self.photo = self.photo.subsample(10, 10)  # Resize the picture (change the subsample values to resize differently)
        self.photo_label = tk.Label(self.title_frame, image=self.photo)
        self.photo_label.pack(side=tk.LEFT, padx=5, pady=5)

        self.title_label = tk.Label(self.title_frame, text="Girlfriend <3", font=("Helvetica", 14, "bold"))
        self.title_label.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.message_frame = tk.Frame(root)
        self.message_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        self.typing_area = tk.Text(root, height=3, wrap=tk.WORD, font=("Helvetica", 13))
        self.typing_area.pack(side=tk.BOTTOM, fill=tk.BOTH, padx=5, pady=5)
        
        self.send_button = tk.Button(root, text="Send", command=self.send_message)
        self.send_button.pack(side=tk.BOTTOM, pady=5)

        self.message_list = []
        self.current_side = "right"
        self.current_row = 0

        self.camera = camera

        
    def send_message(self):
        message = self.typing_area.get("1.0", tk.END).strip()
        print(self.camera.pred)
        if message:
            if self.is_angry(message):
                self.show_custom_popup(message)
            else:
                self.typing_area.delete("1.0", tk.END)

                side = self.current_side
                self.current_side = "left" if self.current_side == "right" else "right"
                
                message_label = tk.Label(self.message_frame, text=message, bg="lightgreen", padx=10, pady=5, 
                                         wraplength=400, font=("Helvetica", 12))
                
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
        popup_window.geometry("600x200")
        popup_window.title("Angry Alert")

        label = tk.Label(popup_window, text="""Hey there big fellow, your heart rate is a bit high and you seem kind of angry,\n are you sure you want to send this? """, 
                         font=("Helvetica", 12), anchor='w')
        label.pack(padx=0, pady=10)

        send_anyway_button = tk.Button(popup_window, text="Send anyway", command=lambda: self.send_message_after_popup(message, popup_window))
        send_anyway_button.pack(side=tk.LEFT, padx=5, pady=5)

        try_again_button = tk.Button(popup_window, text="Let me think about it", command=popup_window.destroy)
        try_again_button.pack(side=tk.LEFT, padx=5, pady=5)

        try_again_button = tk.Button(popup_window, text="See suggested rephrase",command=lambda: [popup_window.destroy(), 
                                                                                                  self.suggested_rephrase()])
        try_again_button.pack(side=tk.LEFT, padx=5, pady=5)        
    

    def suggested_rephrase(self):
        popup_window = tk.Toplevel(self.root)
        popup_window.geometry("800x200")
        popup_window.title("Suggested rephrase")
        rephrase = """I have to say sometimes your responses come across a bit harsh. 
                    Whenever you ask for my assistance,
                    I truly make an effort to help, but it feels like I'm constantly striving to meet your expectations.
                    I would appreciate if we could find a more balanced way to communicate.
                    I'm independent and capable of taking care of myself, so you don't have to feel obligated to pick me up.
                    I'd be happy if next time you'd reply with a little more patience and empathy. """
        label = tk.Label(popup_window, text=rephrase, font=("Helvetica", 12), anchor='w')
        label.pack(padx=0, pady=10)

        send_anyway_button = tk.Button(popup_window, text="Send rehprase", command=lambda: self.send_message_after_popup(rephrase, popup_window))
        send_anyway_button.pack(side=tk.LEFT, padx=5, pady=5)

        try_again_button = tk.Button(popup_window, text="Cancel", command=popup_window.destroy)
        try_again_button.pack(side=tk.LEFT, padx=5, pady=5)
      


    def send_message_after_popup(self, message, popup_window):
        popup_window.destroy()  # Close the pop-up window
        self.typing_area.delete("1.0", tk.END)

        side = self.current_side
        self.current_side = "left" if self.current_side == "right" else "right"
        
        message_label = tk.Label(self.message_frame, text=message, bg="lightgreen", padx=10, pady=5, 
                                 wraplength=400, font=("Helvetica", 12))
        
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
        if polarity < angry_text_threshold and ('Angry' in self.camera.pred_buffer or
                                                'Sad' in self.camera.pred_buffer or
                                                'Disgust' in self.camera.pred_buffer):
            return True
        else:
            return False        
        
# from face_recognition import VideoCamera
# root = tk.Tk()
# WhatsAppLikeApp(root, VideoCamera())
# root.mainloop()