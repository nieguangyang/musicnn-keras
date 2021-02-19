from musicnn.music_nn_keras import MusicNNKeras
from musicnn.example import EXAMPLE
mnn = MusicNNKeras(clip_length=10)  # 10 seconds
audio_file = EXAMPLE
# calculate and visualize taggram
taggram = mnn.audio2taggram(audio_file)
mnn.visualize(taggram)
# extract top k tags
tags, scores = mnn.extract_tags(audio_file, top_k=10)  # take top 10 tags
print(tags)
print(scores)
