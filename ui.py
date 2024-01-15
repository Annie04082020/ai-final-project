import tkinter as tk
import os
from tkinter import *
from tkinter.constants import *
from tkinter import filedialog as fd
from main import predict, train
from record import record_audio
from keras.models import load_model
from pydub import AudioSegment

def transferFile():
    # files                   
    filename = fd.askopenfilename(title='Open file'
                    ,initialdir=r'/home/sholmes/ai-final/project'
                    ,filetypes=[('Audio file','*.mp3')]
                    ,defaultextension='.mp3',multiple=True)     
    num =1                                              
    for item in filename:   
        src = item
        print(item)
        dst = "transferred/transferred_"+str(num)+".wav"
        sound = AudioSegment.from_mp3(src)
        sound.export(dst, format="wav")
        num += 1
    print('Done transferred')
    warning.set('Done transferred')

def importFile():
    filename = fd.askopenfilename(title='Open file'
                    ,initialdir=r'/home/sholmes/ai-final-project'
                    ,filetypes=[('Audio file','*.wav')]
                    ,defaultextension='.wav',multiple=True)
    filenames_str = ", ".join(filename)
    filename_label.set(filenames_str)
    window.update()
    print('selected:',filename)

def prediction():
    data_path = ''
    model = load_model('models_h5/language_identify_model.h5')
    if filename_label.get() == '':
        data_path = 'recorded/recorded_audio.wav'
        if os.path.exists(data_path)==False:
            warning.set('No file selected, please record or select a file.')
            return
    else: 
        data_path = filename_label.get()
    print(data_path)
    predict_file_label.set(data_path)
    predictions, class_ids, class_names=predict(data_path,model)
    file_prediction.set(predictions)
    predict_class_id.set(class_ids)
    predict_class_names.set(class_names)

def training():
    model= train()
    warning.set('Training Done')

window = tk.Tk()
window.title('Language Classification')
window.geometry('1000x600')
window.resizable(False, False)
filename_label = tk.StringVar()
warning = tk.StringVar()
predict_file_label = tk.StringVar()
file_prediction = tk.StringVar()
predict_class_id = tk.StringVar()
predict_class_names = tk.StringVar()

transfer = tk.Button(window, height=3, width =10,text="轉檔.mp3", command = transferFile)
transfer.place(x=100,y=100,anchor=CENTER)

train_btn = tk.Button(window, height=3, width =10,text="訓練", command = training)
train_btn.place(x=100,y=200,anchor=CENTER)

importfile = tk.Button(window, height=3, width =10,text="輸入檔案", command = importFile)
importfile.place(x=100,y=300,anchor=CENTER)

record = tk.Button(window, height=3, width =10,text="錄音", command = record_audio)
record.place(x=100,y=400,anchor=CENTER)

predict_btn = tk.Button(window, height=3, width =10,text="辨識", command = prediction)
predict_btn.place(x=100,y=500,anchor=CENTER)

selected_file=tk.Label(window, textvariable=filename_label, height=7)
filename_label.set('選擇檔案')
selected_file.place(x=450,y=50,anchor=CENTER)

predict_file=tk.Label(window, textvariable=predict_file_label, height=7)
predict_file_label.set('辨識檔案')
predict_file.place(x=450,y=150,anchor=CENTER)

Warning=tk.Label(window, textvariable=warning, height=7)
warning.set('文字提示')
Warning.place(x=450,y=250,anchor=CENTER)

predict_label=tk.Label(window, textvariable=file_prediction, height=7)
file_prediction.set('檔案辨識結果')
predict_label.place(x=450,y=350,anchor=CENTER)

predict_classID = tk.Label(window, textvariable=predict_class_id, height=7)
predict_class_id.set('predict_class_id')
predict_classID.place(x=450,y=450,anchor=CENTER)

predict_classNames = tk.Label(window, textvariable=predict_class_names, height=7)
predict_class_names.set('predict_class_names')
predict_classNames.place(x=450,y=550,anchor=CENTER)

window.mainloop()
