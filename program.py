from __future__ import division, print_function
import sys
import numpy as np
import numpy
import resampy
import tensorflow as tf
import pyaudio
import params
import yamnet as yamnet_model
import numpy
from twilio.rest import Client
from datetime import datetime
import time
global dog_bark
dog_bark = time.time() - 500

Baby_Crying = False


##Get Tensorflow Model 
graph = tf.Graph()
with graph.as_default():
    yamnet = yamnet_model.yamnet_frames_model(params)
    yamnet.load_weights('yamnet.h5')
    yamnet_classes = yamnet_model.class_names('yamnet_class_map.csv')

##Set Paremeters for PyAudio
RATE=16000
RECORD_SECONDS = 3
CHUNKSIZE = 1024
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNKSIZE)

def get_audio_stream():
    stream.start_stream()
    frames = [] # A python-list of chunks(numpy.ndarray)
    print('\n***************\n***************\n***recording***\n***************\n***************\n')
    for _ in range(0, int(RATE / CHUNKSIZE * RECORD_SECONDS)):
        data = stream.read(CHUNKSIZE)
        frames.append(numpy.fromstring(data, dtype=numpy.int16))
    waveform = numpy.hstack(frames)
    waveform = waveform / 32768.0
    stream.stop_stream()
    return waveform

def predict_audio(waveform):
    print('Predicting audio...\n')
    try:
        with graph.as_default():
            scores, _ = yamnet.predict(np.reshape(waveform, [1, -1]), steps=1)
            prediction = np.mean(scores, axis=0)
        prediction_dictionary = {yamnet_classes[i]:prediction[i] for i in range(len(yamnet_classes))}
        if max(prediction_dictionary['Crying, sobbing'], prediction_dictionary['Baby cry, infant cry']) > .1:
            return True
        else:
            dog_noises = ['Domestic animals, pets', 'Dog', 'Bark', 'Yip', 'Howl', 'Bow-wow', 'Growling', 'Whimper (dog)', 'Canidae, dogs, wolves']
            for dog_noise in dog_noises:
                if prediction_dictionary[dog_noise] > .25:
                    dog_bark = time.time()
                    print(f'Dog barked at {dog_bark}')
                    return False
            return False
    except:
        print('Prediction failed...')
        return

def send_message(payload):
    client = Client(account_sid, auth_token)
    message = client.messages \
        .create(
             body=payload,
             messaging_service_sid=messaging_service_sid,
             to='+18608773725')
    
def log_data(timestamp, message):
    time_since_bark = time.time() - dog_bark
    print(time_since_bark)
    if time_since_bark < 150:
        dog_barking_flag = True
    else:
        dog_barking_flag = False
    with open('log.txt', 'a+') as file:
        file.write(f'{str(timestamp)} {message} {dog_barking_flag}\n')

def main(Baby_Crying):
    waveform = get_audio_stream()
    if predict_audio(waveform):
        if Baby_Crying:
            print('Baby still crying... hang in there...')
            return Baby_Crying
        else:
            print('Baby crying present')
            send_message("Someone's Crying!")
            Baby_Crying = True
            log_data(datetime.now(), "Madison started crying.")
            return Baby_Crying
    else:
        if Baby_Crying:
            log_data(datetime.now(), "Madison stopped crying.")
        Baby_Crying = False
        print('No Crying!')
        return Baby_Crying
        
if __name__ == '__main__':
    while True:
        print("\n\nStarting a new round of processing...")
        Baby_Crying = main(Baby_Crying)
        time.sleep(RECORD_SECONDS)
