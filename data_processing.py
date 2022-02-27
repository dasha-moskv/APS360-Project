#Data processing code
#Codes are taken inspiration from Towards AI source [7] in project proposal

import numpy as np
import librosa

sample_rate = 10000 #arbitrary, most english phonemes are 3kHz of bandwidth

#chose so we could use alexnet.features and makes things easier
image_width = 224
image_height = 224

def add_noise(audio_file, gain):
    num_samples = audio_file.shape[0]
    noise = gain*np.random.normal(size=num_samples)
    return audio_file + noise

def load_audio_file(path):
    audio_file, _ = librosa.load(path, sr=sample_rate)
    return audio_file

def change_to_10_s(audio_file):
    target_len = 10*sample_rate
    audio_file = np.concatenate([audio_file]*3, axis = 0)
    audio_file = audio_file[0:target_len]
    return audio_file

def spectrogram(audio_file):
    #compute Mel-scaled spectrogram image
    width = audio_file.shape[0]
    
    spec = librosa.feature.melspectrogram(audio_file,n_mels=image_height, hop_length=int(width))
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
    image_int = image_float_255.astype(numpy.uint8)
    
    return image_int
    
    