import numpy as np
import matplotlib.pyplot as plt

from musicnn.config import CLIP_LENGTH
from musicnn.model import get_mtt_model, get_msd_model
from musicnn.data_processing import audio2x
from musicnn.tags import MTT_TAGS, MSD_TAGS


class MusicNNKeras:
    def __init__(self, clip_length=CLIP_LENGTH, overlap=0.):
        """
        :param clip_length: float, clip length in seconds
        :param overlap: float, overlap in seconds
        """
        # audio processing
        self.clip_length = clip_length
        self.overlap = overlap
        # model
        print("init models")
        self.mtt = get_mtt_model(clip_length)  # model trained on MTT dataset
        self.msd = get_msd_model(clip_length)  # model trained on MSD dataset
        # tags
        self.n_mtt_tags = len(MTT_TAGS)
        self.n_msd_tags = len(MSD_TAGS)
        self.tags = ["%(source)s-%(category)s-%(tag)s" % tag_info for tag_info in MTT_TAGS + MSD_TAGS]

    def audio2x(self, audio_file):
        """
        convert audio file to x, i.e. input to musicnn model
        :param audio_file: str, path to audio file to convert
        :return x: (batch_size, n_timesteps, n_mels, 1) ndarray, input to musicnn model
        """
        x = audio2x(audio_file, self.clip_length, self.overlap)
        return x

    def audio2taggram(self, audio_file):
        """
        audio file -> taggram
        :param audio_file: str, path to audio file
        :return taggram: (n_clips, n_mtt_tags + n_msd_tags), taggram from models respectively trained on
            MTT and MSD datasets
        """
        x = self.audio2x(audio_file)
        # original taggrams from mtt and msd models
        _mtt_taggram = self.mtt.predict(x)
        _msd_taggram = self.msd.predict(x)
        # original taggram -> unified taggram
        n_clips = x.shape[0]
        mtt_taggram = np.zeros((n_clips, self.n_mtt_tags))
        for i, tag_info in enumerate(MTT_TAGS):
            indices = tag_info["indices"]
            for j in indices:
                mtt_taggram[:, i] = np.maximum(mtt_taggram[:, i], _mtt_taggram[:, j])
        msd_taggram = np.zeros((n_clips, self.n_msd_tags))
        for i, tag_info in enumerate(MSD_TAGS):
            indices = tag_info["indices"]
            for j in indices:
                msd_taggram[:, i] = np.maximum(msd_taggram[:, i], _msd_taggram[:, j])
        taggram = np.concatenate((mtt_taggram, msd_taggram), axis=1)
        return taggram

    def visualize(self, taggram, ax=None):
        """
        :param taggram: (n_clips, n_mtt_tags + n_msd_tags), taggram from models respectively trained on
            MTT and MSD datasets
        :param ax: AxesSubplot
        """
        if ax is None:
            given_ax = False
            ax = plt.gca()
        else:
            given_ax = True
        ax.imshow(taggram.T, interpolation=None, aspect="auto")
        ax.title.set_text("taggram")
        # x axis
        n_clips = taggram.shape[0]
        x_ticks = np.arange(n_clips)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_ticks)
        # y axis
        n_tags = len(self.tags)
        y_ticks = np.arange(n_tags)
        y_labels = [tag for tag in self.tags]
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels)
        # show
        if not given_ax:
            plt.show()

    def extract_tags(self, audio_file, top_k=None, ordered=False):
        """
        top n tags from MTT and MSD models
        :param audio_file: str, path to audio file
        :param top_k: int, number of top tags from each model
            if not given, return all tags
            if given, no matter what ordered is, the results will be ordered by score from high to low
        :param ordered: bool, whether rank results by score descend
        :return: (tags, scores)
            tags: list, top k tags from modelss trained respectively on MTT dataset and MSD dataset
            scores: list, corresponding scores
        """
        taggram = self.audio2taggram(audio_file)
        scores = np.mean(taggram, axis=0)
        tag_and_score = [(tag, score) for tag, score in zip(self.tags, scores)]
        if top_k is not None or ordered:  # sort
            tag_and_score.sort(key=lambda ts: ts[1], reverse=True)
        if top_k is not None:
            tag_and_score = tag_and_score[:top_k]
        tags, scores = [], []
        for tag, score in tag_and_score:
            tags.append(tag)
            scores.append(score)
        return tags, scores


def test():
    from musicnn.example import EXAMPLE
    mnn = MusicNNKeras(clip_length=10)
    audio_file = EXAMPLE
    # calculate taggram and visualize
    taggram = mnn.audio2taggram(audio_file)
    mnn.visualize(taggram)
    # extract tags
    tags, scores = mnn.extract_tags(audio_file, top_k=10)
    print(tags)
    print(scores)


if __name__ == "__main__":
    test()
