__network__  
![image](https://github.com/nieguangyang/musicnn-keras/blob/master/musicnn/model.png?raw=true)

__example__  
code  
```
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
```
output  
```
['mtt-genre-rock', 'msd-genre-rock', 'msd-genre-hiphop', 'mtt-timbre-loud', 'mtt-timbre-male', 'mtt-genre-pop', 'mtt-beat-fast', 'mtt-timbre-guitar', 'msd-genre-pop', 'mtt-timbre-vocal']
[0.4091747745319649, 0.2953507282115795, 0.20576066772143045, 0.19431541032261318, 0.18789105393268443, 0.18054986000061035, 0.17712968587875366, 0.17392555210325453, 0.15721004097550004, 0.1306182477209303]
```