import tensorflow as tf
from keras.layers import Input, BatchNormalization, Conv2D, MaxPooling2D, Flatten, Dropout, Dense, Activation
from keras.models import Model

from musicnn.config import N_CLASSES, N_TIMESTEPS, N_MELS, FILTERS, CLIP_LENGTH, SR, HOP_LENGTH
from musicnn.checkpoints import MTT_WEIGHTS, MSD_WEIGHTS


def pad2d(input_tensor, padding):
    """
    zero-pad 2d input tensor
    :param input_tensor: (batch_size, height, width, channels) Tensor, input tensor
    :param padding: tuple, (height_pad, width_pad)
        height_pad: int, pad to top and bottom
        width_pad: int, pad to left and right
    :return output_tensor: (batch_size, padded_height, padded_width, channels) Tensor, output tensor
    """
    height_pad, width_pad = padding
    output_tensor = tf.pad(input_tensor, [[0, 0], [height_pad, height_pad], [width_pad, width_pad], [0, 0]])
    return output_tensor


def midend_block(input_tensor, filters, kernel_size, padding, block_id):
    """
    pad -> conv -> batch norm -> transpose
    :param input_tensor: (batch_size, height, width, channels) Tensor, input tensor
    :param filters: int, number of filters
    :param kernel_size: tuple, (n_timesteps, width)
    :param padding: tuple, (height_pad, width_pad)
        height_pad: int, pad to top and bottom
        width_pad: int, pad to left and right
    :param block_id: int, block id
    :return output_tensor: (batch_size, _height, _width, _channels) Tensor, output tensor
    """
    output_tensor = pad2d(input_tensor, padding)  # zero-pad
    output_tensor = Conv2D(filters, kernel_size, activation="relu", name="midend_conv%d" % block_id)(output_tensor)
    output_tensor = BatchNormalization(name="bn_midend_conv%d" % block_id)(output_tensor)
    output_tensor = tf.transpose(output_tensor, [0, 1, 3, 2])
    return output_tensor


def build_model(n_classes=N_CLASSES, height=N_TIMESTEPS, width=N_MELS, filters=FILTERS):
    """
    :param n_classes: int, number of classes
    :param height: int, number of time steps
    :param width: int, number of frequency bins of mel-spectrogram
    :param filters: tuple or list, (n_frontend_filters, n_midend_filters, n_backend_filters)
        n_frontend_filters: int, number of filters in frontend stage
        n_midend_filters: int, number of filters in midend stage
        n_backend_filters: int, number of filters in backend stage
    :return model: Model, musicnn model
    """
    n_frontend_filters, n_midend_filters, n_backend_filters = filters
    x = Input((height, width, 1))  # (time, frequency, 1)
    # frontend, extract timbral and temporal features
    frontend = BatchNormalization()(x)
    frontend_branches = []
    # frontend, branch 1 and branch 2
    frontend_padded = pad2d(frontend, padding=(3, 0))
    for f in (38, 67):  # number of mel bins (frequency), capture pitch-invariant timbral features
        branch = Conv2D(4 * n_frontend_filters, (7, f), activation="relu")(frontend_padded)
        branch = BatchNormalization()(branch)
        branch = MaxPooling2D((1, width + 1 - f))(branch)
        branch = tf.squeeze(branch, axis=2)
        frontend_branches.append(branch)
    # frontend, branch 3, branch 4 and branch 5
    for t in (128, 64, 32):  # number of time steps, capture temporal energy patterns
        branch = Conv2D(n_frontend_filters, (t, 1), padding="same", activation="relu")(frontend)
        branch = BatchNormalization()(branch)
        branch = MaxPooling2D((1, width))(branch)
        branch = tf.squeeze(branch, axis=2)
        frontend_branches.append(branch)
    frontend = tf.concat(frontend_branches, axis=-1)  # (batch_size, n_timesteps, 11 * n_frontend_filters)
    # midend
    midend = tf.expand_dims(frontend, axis=-1)
    branch1 = midend  # (batch_size, n_timesteps, 11 * n_frontend_filters, 1)
    branch2 = midend_block(branch1, filters=n_midend_filters, kernel_size=(7, 11 * n_frontend_filters), padding=(3, 0), block_id=1)  # (batch_size, n_timesteps, n_midend_filters, 1)
    branch3 = midend_block(branch2, filters=n_midend_filters, kernel_size=(7, n_midend_filters), padding=(3, 0), block_id=2)
    branch3 = tf.add(branch3, branch2)  # (batch_size, n_timesteps, n_midend_filters, 1)
    branch4 = midend_block(branch3, filters=n_midend_filters, kernel_size=(7, n_midend_filters), padding=(3, 0), block_id=3)
    branch4 = tf.add(branch4, branch3)  # (batch_size, n_timesteps, n_midend_filters, 1)
    midend_branches = [branch1, branch2, branch3, branch4]
    midend = tf.concat(midend_branches, axis=2)  # (batch_size, n_timesteps, 11 * n_frontend_filters + 3 * n_midend_filters)
    # backend
    backend_max = tf.reduce_max(midend, axis=1)
    backend_mean = tf.reduce_mean(midend, axis=1)
    backend = tf.concat([backend_max, backend_mean], axis=-1)  # (batch_size, 11 * n_frontend_filters + 3 * n_midend_filters, 2)
    backend = Flatten()(backend)
    backend = BatchNormalization(name="bn_flatpool")(backend)
    backend = Dropout(0.5)(backend)
    backend = Dense(n_backend_filters, activation="relu")(backend)
    backend = BatchNormalization(name="bn_dense")(backend)
    backend = Dropout(0.5)(backend)
    backend = Dense(n_classes)(backend)
    y = Activation("sigmoid")(backend)
    model = Model(x, y)
    return model


def get_mtt_model(clip_length=CLIP_LENGTH):
    """
    :param clip_length: float, clip length in seconds
    :return mtt: Model, model pre-trained on MTT dataset
    """
    n_timesteps = int(clip_length * SR // HOP_LENGTH)
    mtt = build_model(height=n_timesteps)
    print("load weights from %s" % MTT_WEIGHTS)
    mtt.load_weights(MTT_WEIGHTS)
    return mtt


def get_msd_model(clip_length=CLIP_LENGTH):
    """
    :param clip_length: float, clip length in seconds
    :return msd: Model, model pre-trained on MSD dataset
    """
    n_timesteps = int(clip_length * SR // HOP_LENGTH)
    msd = build_model(height=n_timesteps)
    print("load weights from %s" % MSD_WEIGHTS)
    msd.load_weights(MSD_WEIGHTS)
    return msd


def test():
    # from keras.utils import plot_model
    # mtt = get_mtt_model()
    # plot_model(mtt, "./model.png", show_shapes=True)
    mtt = get_mtt_model()
    mtt.summary()


if __name__ == "__main__":
    test()
