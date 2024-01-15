import tkinter as tk
from tkinter.constants import *
from tkinter import filedialog as fd
from main import predict
from record import record_audio

def importFile(event = None):
    filename = fd.askopenfilename()
    filetypes = (
        ('text files', '*.txt'),
        ('All files', '*.*')
    )
    print('selected:',filename)

window = tk.Tk()
window.title('Language Classification')
window.geometry('1000x800')
window.resizable(False, False)
importfile = tk.Button(window, text="輸入檔案", command = 'importFile')
importfile.place(x=190,y=200,anchor=CENTER)
importfile.pack(expand=True)
# record = tk.Button(window, text="錄音", command = 'record_audio')
# record.place(x=250,y=200,anchor=CENTER)
# record.pack(expand=True)

window.mainloop()
