import sys
import librosa
import scipy.signal as signal
import numpy as np
from PIL import Image
import PIL.ImageOps


class PreProcessing:
    def __init__(self, filename: str, samplerate: int = 8320, medfilter: int = 5, start_second: int = None, end_second: int = None):
        '''
        Processes audio stream from weather satellites.
        :param filename: absolute filepath and name
        :param samplerate: 8320 recommended (2080 * 4)
        :param start_second: allows for faster processing
        :param end_second: allows for faster processing
        '''

        self.__start = start_second
        self.__end = end_second
        self.target_rate = samplerate

        print("Loading Audio...")
        self.original_audio, self.original_rate = self.__load_file(filename)

        print("Adjusting Audio...")
        adjusted_audio = self.__adjust_audio(self.original_audio, self.original_rate, self.target_rate, medfilter)

        print("Digitizing Values...")
        self.raw_signal = self.__digitize(adjusted_audio)

        print("Saving Raw Image and Data...")
        self.save_original_image(self.raw_signal, samplerate)

        print("Synchronizing Frames...")
        self.frames = self.__reshape(self.raw_signal)

        self.save_image(self.frames, "frames.png")

        print("Seperating Frames....")
        a, b = self.__channel_cropper("F:/Python Projects/Satellite/frames.png")
        # lTODO: get a and b into their own PNG files!

        self.__colourize(a, b)

    def __load_file(self, filepath):
        '''

        :param filepath: absolute location of file
        :return: list of stream, and sample rate of stream.
        '''
        y, sr = librosa.load(filepath, mono=True)
        if self.__start is not None and self.__end is not None:
            y = y[self.__start * sr:self.__end * sr]
        return y, sr

    def __adjust_audio(self, s, orig_rate, targ_rate, medfilter):
        y_norm = librosa.util.normalize(s)
        y_resampled = librosa.resample(y_norm, orig_sr=orig_rate, target_sr=targ_rate)
        y_renormalized = librosa.util.normalize(y_resampled)

        analytical_signal = signal.hilbert(y_renormalized)
        analytical_signal = signal.medfilt(np.abs(analytical_signal), medfilter)

        return np.abs(analytical_signal)

    def __digitize(self, s, plow=0.5, phigh=99.5):
        '''
        Convert signal to numbers between 0 and 255.
        :param s: signal data
        :param plow: lower bound (0) rank order
        :param phigh: high bound (255) rank order
        :return: matrix of digitized data
        '''
        (low, high) = np.percentile(s, (plow, phigh))
        delta = high - low
        data = np.round(255 * (s - low) / delta)
        data[data < 0] = 0
        data[data > 255] = 255
        return data.astype(np.uint8)

    def __reshape(self, s):
        '''
        Find sync frames and reshape the 1D signal into a 2D image
        Finds the sync A frame by looking at the maximum values of the
        cross-correlation between the signal and a hardcoded sync A frame.

        Expected distance between sync A frames is 2080 samples, but with
        small variations because of Doppler effect
        :param s: 1D signal matrix
        :return: 2d image matrix
        '''

        # syncA = 39 pixels * 2 = 78 + Space A = 47 pixels *2 = 94

        syncA_start =   [0] * 8
        syncA_pattern = ([128] + [255] * 5 + [128] + [0] * 3) * 7
        syncA_end = [255] * 94
        syncA = syncA_start + syncA_pattern + syncA_end
        # 93 + 8 + 70

        # index, correlation
        peaks = [(0, 0)]

        # expected distance is 4160 (2080 for a frame * 2 = 4160 * 2 = 8320 framerate)
        mindistance = 4100

        # shift values down to get meaningful correlations values
        signalshifted = [x-128 for x in s]
        syncA = [x-128 for x in syncA]

        for i in range(len(s) - len(syncA)):
            corr = np.dot(syncA, signalshifted[i : i + len(syncA)])

            # if previous peak is too far, keep it and add this value
            # to the list as a new peak
            if i - peaks[-1][0] > mindistance:
                peaks.append((i, corr))

            # TODO: if peak not found, then must assume previous peak + 2080?

            # else if this value is bigger than the previous max, set this one
            elif corr > peaks[-1][1]:
                peaks[-1] = (i, corr)

        matrixA = []

        for i in range(len(peaks) - 1):
            matrixA.append(s[peaks[i][0] : peaks[i][0] + 4160])

        self.save_array(peaks, "peaks.csv")
        return np.array(matrixA).astype(np.uint8)

    def save_image(self, image, name):
        img = Image.fromarray(image)
        img.save(name)
        img.show()

    def save_array(self, data, name):
        np.savetxt(name, data, fmt='%i', delimiter=',')

    def save_original_image(self, s, rate):
        frame_width = int(rate * 0.5)
        w, h = frame_width, s.shape[0]//frame_width
        s = s[0:w * h]
        reshaped = s.reshape(h, w)
        img = Image.fromarray(reshaped)
        img.save("raw.png")
        img.show()

        self.save_array(reshaped, "raw.csv")

    def __channel_cropper(self, filename):
        # These were shamelessly taken from aptdec
        factor = int(self.target_rate / 2080)
        APT_SYNC_WIDTH = 39 * factor
        APT_SPC_WIDTH = 47 * factor
        APT_TELE_WIDTH = 45 * factor
        APT_FRAME_LEN = 128
        APT_CH_WIDTH = 909 * factor

        i = Image.open(filename).convert("L")
        #iar = i.load()
        xsize, ysize = i.size
        #cha = Image.new('RGB', (APT_CH_WIDTH, ysize))
        #chb = Image.new('RGB', (APT_CH_WIDTH, ysize))
        (left, upper, right, lower) = (
        APT_SYNC_WIDTH + APT_SPC_WIDTH, 1, APT_SYNC_WIDTH + APT_SPC_WIDTH + APT_CH_WIDTH - 10, ysize)
        cha = i.crop((left, upper, right, lower))
        # cha.show()
        (left, upper, right, lower) = (
        APT_SYNC_WIDTH + APT_SPC_WIDTH + APT_CH_WIDTH + APT_SYNC_WIDTH + APT_SPC_WIDTH + APT_TELE_WIDTH, 1,
        APT_SYNC_WIDTH + APT_SPC_WIDTH + APT_CH_WIDTH + APT_SYNC_WIDTH + APT_SPC_WIDTH + APT_CH_WIDTH + 10, ysize)

        chb = i.crop((left, upper, right, lower))

        # chb.show()
        print(APT_CH_WIDTH, ysize)
        cha = cha.resize((APT_CH_WIDTH, ysize), Image.ANTIALIAS)
        chb = chb.resize((APT_CH_WIDTH, ysize), Image.ANTIALIAS)

        return cha, chb

    def __colourize(self, ch2, ch4):
        args = list(sys.argv)

        try:
            args[1]
        except:
            args.append("-noir")

        try:
            args[2]
        except:
            if args[1] == "-boost":
                args.append("-boost")
            else:
                args.append("-noboost")

        # images
        #ch2 = Image.open("F:/Python Projects/Satellite/cha.png").convert("L")
        #ch4 = Image.open("F:/Python Projects/Satellite/chb.png").convert("L")

        # ch4 = PIL.ImageOps.invert(ch4)
        # ch4 = PIL.ImageOps.autocontrast(ch4)
        ch2 = PIL.ImageOps.autocontrast(ch2)

        # Variables
        deyellowfactor = 100
        black_pixels = []
        ch2_blend = 1
        ch4_blend = 200
        ch2_boost = 1.5
        ch4_boost = 0.5

        # convert to arrays
        ch2_array = ch2.load()
        ch4_array = ch4.load()

        # sizes
        xsize, ysize = ch2.size

        # output
        outimg = Image.new('RGB', (xsize, ysize))

        def change_contrast(img, level):
            factor = (259 * (level + 255)) / (255 * (259 - level))

            def contrast(c):
                return 128 + factor * (c - 128)

            return img.point(contrast)

        # main loop
        """
        for y in range(ysize):
            for x in range(xsize):

                ch4.putpixel((x,y),(ch2_array[x,y],ch2_array[x,y],int(1*ch4_array[x,y])))
        """
        if args[1] == "-ir":
            for y in range(ysize):
                for x in range(xsize):
                    if ch2_array[x, y] - ch4_array[x, y] <= -50:
                        outimg.putpixel((x, y), (ch4_array[x, y], ch4_array[x, y], int(1 * ch4_array[x, y])))
                    else:
                        outimg.putpixel((x, y), (ch2_array[x, y], ch2_array[x, y], int(1 * ch4_array[x, y])))
        elif args[1] == "-ir_night":
            for y in range(ysize):
                for x in range(xsize):
                    if ch2_array[x, y] - ch4_array[x, y] <= -40:
                        outimg.putpixel((x, y), (round(ch2_array[x, y] * 0.8 + ch4_array[x, y] * 0.2),
                                                 round(ch2_array[x, y] * 0.8 + ch4_array[x, y] * 0.2),
                                                 round(ch2_array[x, y] * 0.8 + ch4_array[x, y] * 0.2)))
                        # black_pixels.append([x,y])
                    else:
                        outimg.putpixel((x, y), (ch2_array[x, y], ch2_array[x, y], int(1 * ch4_array[x, y])))

        elif args[1] == "-sunset":
            for y in range(ysize):
                for x in range(xsize):
                    if ch2_array[x, y] - ch4_array[x, y] <= -40:
                        outimg.putpixel((x, y), (round(ch2_array[x, y] * ch2_blend), round(ch2_array[x, y] * ch2_blend),
                                                 round(ch4_array[x, y] * ch2_array[x, y] / ch4_blend)))
                        # black_pixels.append([x,y])
                    else:
                        outimg.putpixel((x, y), (ch2_array[x, y], ch2_array[x, y], int(1 * ch4_array[x, y])))

        else:
            for y in range(ysize):
                for x in range(xsize):
                    outimg.putpixel((x, y), (ch2_array[x, y], ch2_array[x, y], int(1 * ch4_array[x, y])))
        if args[2] == "-boost":
            outimg_array = outimg.load()
            for y in range(ysize):
                for x in range(xsize):
                    outimg.putpixel((x, y), (
                    round(outimg_array[x, y][0] * ch2_boost), round(outimg_array[x, y][1] * ch2_boost),
                    round(outimg_array[x, y][2] * ch4_boost)))

        outimg = PIL.ImageOps.autocontrast(outimg)
        # outimg = change_contrast(outimg,100)

        outimg.show()
        outimg.save("color.png")

