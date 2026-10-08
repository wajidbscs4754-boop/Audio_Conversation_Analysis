import librosa

audio_path = "data/conversation.wav"

audio, sample_rate = librosa.load(audio_path, sr=None)

duration = librosa.get_duration(y=audio, sr=sample_rate)

print("Audio Information")
print("-----------------")
print(f"Sample rate: {sample_rate} Hz")
print(f"Duration: {duration:.2f} seconds")
print(f"Total samples: {len(audio)}")