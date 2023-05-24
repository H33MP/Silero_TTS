import os
import torch
import sys
import argparse


'''
Для установки torch нужно выполнить команду:

        pip install torch
'''




example_text = sys.argv[1]
sample_rate = 48000
speaker='kseniya'



# os.startfile('test.wav')

def Init():

    arg = argparse.ArgumentParser()

    arg.add_argument('-text', help='Ввод текста для озвучки')
    arg.add_argument('-voice', help='Выбор голоса. По умолчанию kseniya',
                     choices=['aidar', 'baya', 'kseniya', 'xenia', 'eugene', 'random'],
                     default='kseniya')

    args = arg.parse_args()
    print(args)

def Torching(text_for_speach):

    device = torch.device('cpu')
    torch.set_num_threads(4)
    local_file = 'model.pt'

    if not os.path.isfile(local_file):
        torch.hub.download_url_to_file('https://models.silero.ai/models/tts/ru/v3_1_ru.pt',
                                       local_file)

    model = torch.package.PackageImporter(local_file).load_pickle("tts_models", "model")
    model.to(device)

    audio_paths = model.save_wav(text=text_for_speach,
                                 speaker=speaker,
                                 sample_rate=sample_rate)

Init()