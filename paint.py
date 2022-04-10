import tkinter as tk
import subprocess
import os
import io
from PIL import Image

class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.line_start = None
        self.canvas = tk.Canvas(self, width=300, height=300, bg="white")
        self.canvas.bind("<B1-Motion>", lambda e: self.draw(e.x, e.y))
        self.button = tk.Button(self, text="save",
                                command=self.save)
        self.canvas.pack()
        self.button.pack(pady=10)

    def draw(self, x, y):
        x1, y1 = (x-3), (y-3)
        x2, y2 = (x+3), (y+3)
        color = "black"
        self.canvas.create_oval(x1, y1, x2, y2, fill=color, outline=color)

    def save(self):
        ps = self.canvas.postscript(colormode='color')
        img = Image.open(io.BytesIO(ps.encode('utf-8')))
        img.save('number.jpg')

app = App()
app.mainloop()