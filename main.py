import sat_audio.preprocessing

p = sat_audio.preprocessing.PreProcessing(
    'F:/Python Projects/Satellite/recordings/audio_137103012Hz_20-09-46_12-08-2022.wav', 8320)

print(p.frameA)
p.save_array(p.frameA, 'frameA.csv')
p.save_image(p.frameA, 'frameA.png')

