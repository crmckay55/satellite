import librosa
import scipy.signal as signal
import numpy as np
from PIL import Image


class PreProcessing():
    def __init__(self, filename: str, samplerate: int = 11025):
        self.__source_file = filename
        self.target_rate = samplerate
        self.original_audio, self.original_rate = self.__load_file(self.__source_file)
        self.__adjusted_audio = self.__adjust_audio()
        self.__luminosity_data = self.__convert_luminosity()

    def __load_file(self, f):
        y, sr = librosa.load(f, mono=True)
        return y, sr

    def __adjust_audio(self):
        y_norm = librosa.util.normalize(self.original_audio)
        y_resampled = librosa.resample(y_norm, orig_sr=self.original_rate, target_sr=self.target_rate)
        y_renormalized = librosa.util.normalize(y_resampled)

        analytical_signal = signal.hilbert(y_renormalized)
        return np.abs(analytical_signal)

    def __convert_luminosity(self):
        divisor = max(self.__adjusted_audio) / 255
        frame_width = int(0.5*self.target_rate)
        w, h = frame_width, self.__adjusted_audio.shape[0] // frame_width
        data_am = self.__adjusted_audio[0:w * h]
        reshaped = np.reshape(data_am, (h, w))
        vfunc = np.vectorize(self.__normalize_lum)
        return vfunc(reshaped, divisor)

    def __normalize_lum(self, value, divisor):
        lum = int(value//divisor)
        if lum < 0: lum = 0
        if lum > 255: lum = 255
        return lum

    def save_image(self):
        img = Image.fromarray(self.__luminosity_data)
        img.save("test.png")
        img.show()


