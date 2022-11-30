import satellite.preprocessing

p = satellite.preprocessing.PreProcessing(
    'F:/Python Projects/Satellite/recordings/audio_137103012Hz_20-09-46_12-08-2022.wav', 8320)

print(p.frames)
p.save_array(p.frames, 'frames.csv')


