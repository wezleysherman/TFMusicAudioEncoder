# TFMusicAudioEncoder
An Autoencoder for WAV files


## Setup
1. Create a folder named 'audio_wav' and a folder named 'output'
2. Put all WAV files for training in the audio_wav folder
3. Open up your terminal within the folder and run `python3 encoder.py`

## Tips
- If you're only looking at a single batch of songs, it'd be wise to move the
```ch1_song, ch2_song, sample_rate = next_batch(i, batch_size, sess)```
call outside of the training loops. This way it's only called once.

- If you find yourself running out of memory too quickly, reduce the songs_per_batch, and the node sizes for the input layers.

