import tkinter as tk
import os
from tkinter import *
from tkinter.constants import *
from tkinter import filedialog as fd
from main import predict, train
from record import record_audio
from keras.models import load_model
from pydub import AudioSegment

model_path = ''
labels = ['de', 'en', 'es', 'fr', 'it', 'jp', 'se','tw']

def transferFile():
    # files                   
    filename = fd.askopenfilename(title='Open file'
                    ,initialdir= '/home/sholmes/ai-final-project'
                    ,filetypes=[('Audio file','*.mp3')]
                    ,defaultextension='.mp3', multiple=True)     
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
                    ,initialdir= '/home/sholmes/ai-final-project'
                    ,filetypes=[('Audio file','*.wav')]
                    ,defaultextension='.wav',multiple=True)
    filenames_str = ", ".join(filename)
    filename_label.set(filenames_str)
    print('selected:',filename)

def selectModel():
    filename = fd.askopenfilename(title='Open file'
                    ,initialdir= '/home/sholmes/ai-final-project/models_h5'
                    ,filetypes=[('Keras model file','*.h5')]
                    ,defaultextension='.h5',multiple=True)
    model_path = filename    
    chosen_model.set(model_path)
    print('selected model:',model_path)

def prediction(model_load = model_path):
    data_path = ''
    model_load = 'models_h5/language_identify_model.h5'
    print(model_load)
    model = load_model(model_load)
    if filename_label.get() == '':
        data_path = 'recorded/recorded_audio.wav'
        if os.path.exists(data_path)==False:
            warning.set('No file selected, please record or select a file.')
            return
    else: 
        data_path = filename_label.get()
    print(data_path)
    predict_file_label.set(data_path)
    prediction = predict(data_path,model)
    predict_result.set(prediction)
    # predictions, class_ids, class_names=predict(data_path,model)
    # file_prediction.set(predictions)
    # predict_class_id.set(class_ids)
    # predict_class_names.set(class_names)

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
chosen_model = tk.StringVar()
predict_result = tk.StringVar()
default_labels = tk.StringVar()

transfer = tk.Button(window, height=3, width =10,text="轉檔.mp3", command = transferFile)
transfer.place(x=100,y=100,anchor=CENTER)

ch_model = tk.Button(window, height=3, width =10,text="選擇模型", command = selectModel)
ch_model.place(x=700,y=500,anchor=CENTER)

train_btn = tk.Button(window, height=3, width =10,text="訓練", command = training)
train_btn.place(x=100,y=200,anchor=CENTER)

importfile = tk.Button(window, height=3, width =10,text="輸入檔案", command = importFile)
importfile.place(x=100,y=300,anchor=CENTER)

record = tk.Button(window, height=3, width =10,text="錄音", command = record_audio)
record.place(x=100,y=400,anchor=CENTER)

predict_btn = tk.Button(window, height=3, width =10,text="辨識", command = prediction)
predict_btn.place(x=100,y=500,anchor=CENTER)

selected_file=tk.Label(window, textvariable=filename_label, height=3)
filename_label.set('選擇檔案')
selected_file.place(x=450,y=50,anchor=CENTER)

selected_model=tk.Label(window, textvariable=chosen_model, height=3)
chosen_model.set('選擇網路模型檔案')
selected_model.place(x=450,y=100,anchor=CENTER)

predict_file=tk.Label(window, textvariable=predict_file_label, height=3)
predict_file_label.set('辨識檔案')
predict_file.place(x=450,y=150,anchor=CENTER)

Warning=tk.Label(window, textvariable=warning, height=3)
warning.set('文字提示')
Warning.place(x=450,y=200,anchor=CENTER)

predict_label=tk.Label(window, textvariable=file_prediction, height=3)
file_prediction.set('file_prediction')
predict_label.place(x=450,y=250,anchor=CENTER)

predict_classID = tk.Label(window, textvariable=predict_class_id, height=3)
predict_class_id.set('predict_class_id')
predict_classID.place(x=450,y=300,anchor=CENTER)

predict_classNames = tk.Label(window, textvariable=predict_class_names, height=3)
predict_class_names.set('predict_class_names')
predict_classNames.place(x=450,y=350,anchor=CENTER)

predict_results = tk.Label(window, textvariable=predict_result, height=3)
predict_result.set('prediction_result')
predict_results.place(x=450,y=400,anchor=CENTER)

label_display = tk.Label(window, textvariable = default_labels, height=3)
default_labels.set(labels)
label_display.place(x=450,y=500,anchor=CENTER)

window.mainloop()
