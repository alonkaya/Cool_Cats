from tkinter import *
from textblob import TextBlob


def submit():
    input_text = text.get("1.0", END)
    blob_text = TextBlob(input_text)
    sentiment = blob_text.sentiment 
    polarity = sentiment.polarity

    if polarity < -0.6:
        label.config(text= "Hey there big fellow :) I noticed you're angry\n would you like to take a break and send this message later?")
    else:
        label.config(text=input_text)  # Set the text of the label to the input_text

window = Tk()
text = Text(window, font=("Ink Free", 25), height=8, width=30)
text.pack()

button = Button(window, text="Submit", command=submit)
button.pack()

label = Label(window, font=("Ink Free", 25), height=8, width=30, wraplength=300)
label.pack()

window.mainloop()
