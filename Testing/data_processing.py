#Data processing code
#Codes are taken inspiration from Towards AI source [7] in project proposal
#to download mp3 files from tatoeba, use https://audio.tatoeba.org/sentences/{lang}/{id}.mp3

import os
import numpy as np
import librosa
from librosa.display import waveshow
from pyparsing import srange
import soundfile
#import imageio
import matplotlib
import matplotlib.pyplot as plt
from torch import fmax

sr = 10000 #arbitrary, most english phonemes are 3kHz of bandwidth

#chose so we could use alexnet.features and makes things easier
image_width = 224
image_height = 224

def add_noise(audio_file, gain):
    num_samples = audio_file.shape[0]
    noise = gain*np.random.normal(size=num_samples)
    return audio_file + noise

#for some reason, this only works with .wav files
def load_audio_file(path):
    audio_file, sample_rate = librosa.load(path)
    return audio_file, sample_rate

def change_to_10_s(audio_file, sample_rate):
    target_len = 10*sample_rate
    audio_file = np.concatenate([audio_file]*10, axis = 0)
    audio_file = audio_file[0:target_len]
    return audio_file

#ignore the warnings when it does melspectrogram
def spectrogram(audio_file, sample_rate):
    #compute Mel-scaled spectrogram image
    #print(audio_file.shape)
    
    spec = librosa.feature.melspectrogram(audio_file, sr=sample_rate, n_mels=image_height)
    image = librosa.core.power_to_db(spec) #log image
    #convert to np matrix
    image_np = np.asmatrix(image)

    #normalize and scale
    image_np_scaled_temp = (image_np - np.min(image_np))
    image_np_scaled = image_np_scaled_temp/np.max(image_np_scaled_temp)
    return image_np_scaled[:, 0:image_width]

def to_png(image_float):
    #range (0,1) to (0,255)
    image_float_255 = image_float*255.0
    image_int = image_float_255.astype(np.uint8)
    
    return image_int
    
def augment_audio_file(source_path):
    audio_segment, sample_rate = load_audio_file(source_path)
    audio_segment_with_noise = add_noise(audio_segment, 0.005)
    path = os.path.splitext(source_path)[0] + '_augmented_noise.wav'
    soundfile.write(path, audio_segment_with_noise, sample_rate)

def audio_to_image(audio_file, sample_rate, destination_folder):
    image_file = destination_folder + '.png'
    audio, sample_rate = load_audio_file(audio_file)
    audio_fixed = change_to_10_s(audio, sample_rate)
    if np.count_nonzero(audio_fixed) != 0:
        spectro = spectrogram(audio_fixed, sr)
        spectro_img = to_png(spectro)
        print("Image scales: ", spectro_img.shape)
        #spectro_tuple = (spectro_img,spectro_img,spectro_img)
        #spectro_RGB = np.vstack(spectro_tuple)
        #imageio.imwrite(image_file, spectro_RGB)
        matplotlib.image.imsave(image_file, spectro_img)
        return spectro_img
    
###Testing###

#augment a sample .wav file
path1 = './Languages/English_audio/English_test_file.wav'
#augment_audio_file(path)

#convert to spectrogram
path2 = './Languages/English_audio/English_test_file_augmented_noise.wav'
#audio_to_image(path, './Languages/English/test')
'''
audio, sample = load_audio_file(path1)
waveshow(audio, sr=sample)
plt.title('Raw Audio File')
#plt.show()

audio_fixed = change_to_10_s(audio, sample)
waveshow(audio_fixed, sr=sr)
plt.title('Raw Audio File fixed')
#plt.show()


spectro = spectrogram(audio_fixed,sr)
spectro = to_png(spectro)
#spectro = librosa.display.specshow(audio_fixed, sr=sr)
#spectro.savefig('test.png')

plt.imshow(spectro, origin='lower', aspect='auto')
plt.title('Spectrogram')
plt.show()
print(spectro.shape) #for now the shape is arbitrary, we need to make it so it's consistent throughout all data
'''
spectro = audio_to_image(path2, sr, './Languages/English/test')
