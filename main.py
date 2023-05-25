import os
import torch
import sys
import argparse
from datetime import datetime


'''
Для установки torch нужно выполнить команду:

        pip install torch
'''


def Init():

    arg = argparse.ArgumentParser()

    arg.add_argument('-text', help='Ввод текста для озвучки', required=True)
    arg.add_argument('-voice', help='Выбор голоса. По умолчанию kseniya',
                     choices=['aidar', 'baya', 'kseniya', 'xenia', 'eugene', 'random'],
                     default='kseniya')
    arg.add_argument('-rate', help='установить качество звука. По умолчанию 48000',
                     choices=[8000, 24000, 48000], default=48000)
    arg.add_argument('-name', help='Имя сохраняемого файла', default=datetime.strftime(datetime.now(), '%H.%M.%S_%d.%m.%Y'))

    args = arg.parse_args()

    Torching(text_for_speach=args.text, voice=args.voice, rate=args.rate, file_name=f'{args.name}.wav')

def Torching(*, text_for_speach, voice, rate, file_name):

    device = torch.device('cpu')
    torch.set_num_threads(4)
    local_file = 'model.pt'

    if not os.path.isfile(local_file):
        torch.hub.download_url_to_file('https://models.silero.ai/models/tts/ru/v3_1_ru.pt',
                                       local_file)

    model = torch.package.PackageImporter(local_file).load_pickle("tts_models", "model")
    model.to(device)

    audio_paths = model.save_wav(text=text_for_speach,
                                 speaker=voice,
                                 sample_rate=rate,
                                 audio_path=file_name)

Init()