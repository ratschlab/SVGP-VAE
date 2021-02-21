"""
Utils functions for the SPRITES experiment.
"""

SPRITES_repo_path = '../sprites/'

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
import pickle
import random
from sklearn.decomposition import PCA
import scipy
import glob
import sys
from pathlib import Path

sys.path.append(SPRITES_repo_path)
from load_sprites import sprites_act

tfk = tfp.math.psd_kernels


def group_by_characters(df):
    """
    Gathers all actions for each character.

    :return: dict with indices of all actions per character
    """

    character_styles = [list(np.nonzero(df[x][0])[1]) for x in range(len(df))]

    totals = {}
    for i, k in enumerate(character_styles):
        if str(k) in totals.keys():
            totals[str(k)][0] += 1
            totals[str(k)][1].append(i)
        else:
            totals[str(k)] = [1, [i]]

    return totals


def preprocess_sprite_SVGPVAE(path, N_frames_train, N_actions=9, T=8):
    """
    Reshape Sprites data into format that SVGPVAE models expects (frames, aux_data).

    :param path: path to original .npy Sprites files
    :param N_frames_train: number of frames (out of 72) per character to use for training
    :param N_actions: number of actions per character
    :param T: number of frames per action

    :return: train_frames (1000 * N_frames_train, 64, 64, 3),
             train_aux_data (1000 * N_frames_train, 2),
             test_action_frames (1000 * (72-N_frames_train), 64, 64, 3),
             test_action_aux_data (1000 * (72-N_frames_train), 2),
             test_character_frames (296 * 72, 64, 64, 3),
             test_character_aux_data (296 * 72, 2)
    """
    assert 0 < N_frames_train <= 72
    flatten = lambda l: [item for sublist in l for item in sublist]

    frames_per_char = N_actions * T

    X_train, X_test, A_train, A_test, D_train, D_test = sprites_act(path, return_labels=True)

    # group by on character style for train data
    characters_train = group_by_characters(A_train)

    # construct train and test_action datasets
    train_frames, train_aux_data = [], []
    test_action_frames, test_action_aux_data = [], []

    for i, character_frames in enumerate(characters_train.values()):
        ids = character_frames[1]
        frames = X_train[ids].reshape(-1, 64, 64, 3)

        # sample N_frames_train frames uniformly at random for each character
        train_ids = random.sample(list(range(frames_per_char)), N_frames_train)
        test_ids = list(set(list(range(frames_per_char))) - set(train_ids))

        train_frames.append(frames[train_ids])
        test_action_frames.append(frames[test_ids])

        # construct auxiliary data
        actions = [np.nonzero(D_train[i][0])[0][0] for i in ids]
        actions = np.array(flatten([[i for i in range(action * T, (action + 1) * T)] for action in actions]))

        actions_train = actions[train_ids]
        actions_test = actions[test_ids]

        train_aux_data_i = np.stack(([i] * len(actions_train), actions_train), axis=-1)
        train_aux_data.append(train_aux_data_i)

        test_aux_data_i = np.stack(([i] * len(actions_test), actions_test), axis=-1)
        test_action_aux_data.append(test_aux_data_i)

    # group by on character style for test data
    characters_test = group_by_characters(A_test)

    # construct test_character_style dataset
    test_char_frames, test_char_aux_data = [], []

    for i, character_frames in enumerate(characters_test.values()):
        ids = character_frames[1]
        frames = X_test[ids].reshape(-1, 64, 64, 3)
        test_char_frames.append(frames)

        # construct auxiliary data
        actions = [np.nonzero(D_test[i][0])[0][0] for i in ids]
        actions = np.array(flatten([[i for i in range(action * T, (action + 1) * T)] for action in actions]))

        test_aux_data_i = np.stack(([i] * len(actions), actions), axis=-1)
        test_char_aux_data.append(test_aux_data_i)

    train_frames = np.concatenate(train_frames)
    train_aux_data = np.concatenate(train_aux_data)
    test_action_frames = np.concatenate(test_action_frames)
    test_action_aux_data = np.concatenate(test_action_aux_data)
    test_char_frames = np.concatenate(test_char_frames)
    test_char_aux_data = np.concatenate(test_char_aux_data)

    print("Train frames dim: {}".format(train_frames.shape))
    print("Train aux data dim: {}".format(train_aux_data.shape))
    print("Test action frames dim: {}".format(test_action_frames.shape))
    print("Test action aux data dim: {}".format(test_action_aux_data.shape))
    print("Test char style frames dim: {}".format(test_char_frames.shape))
    print("Test char style aux data dim: {}".format(test_char_aux_data.shape))

    return train_frames, train_aux_data, test_action_frames, test_action_aux_data, test_char_frames, test_char_aux_data


def npy_to_tfrecords(arr_frames, arr_aux_data, output_file):
    """
    Write records to a tfrecords files (so that later on we do not have to load the entire dataset to memory at once).

    :param arr_frames:
    :param arr_aux_data:
    :param output_file:

    """
    writer = tf.python_io.TFRecordWriter(output_file)

    # Loop through all the features you want to write
    for i in range(len(arr_frames)):
        # Feature contains a map of string to feature proto objects
        feature = {}
        feature['frame'] = tf.train.Feature(float_list=tf.train.FloatList(value=arr_frames[i].flatten()))
        feature['character_ID'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[arr_aux_data[i, 0]]))
        feature['action_ID'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[arr_aux_data[i, 1]]))

        # Construct the Example proto object
        example = tf.train.Example(features=tf.train.Features(feature=feature))

        # Serialize the example to a string
        serialized = example.SerializeToString()

        # write the serialized objec to the disk
        writer.write(serialized)
    writer.close()


def import_sprites(batch_size, sprites_path='SPRITES data/', batch_size_test_char=576):
    """

    Support for loading of data and batching via tf.data.Dataset API.

    :param batch_size:
    :param sprites_path:
    :param batch_size_test_char: batch size for prediction pipeline for test_character dataset. 576 is chosen since
                                21312 % 576 == 0, so that implemenation of splitting of test batch into
                                context and target frames is easier.

    :return: train_iterator,
             test_action_iterator,
             test_character_iterator
    """

    assert batch_size_test_char % 72 == 0

    filenames_train = glob.glob(sprites_path + "train/*.tfrecord")
    filenames_test_action = glob.glob(sprites_path + "test_action/*.tfrecord")
    filenames_test_char = glob.glob(sprites_path + "test_character/*.tfrecord")

    dataset_train = tf.data.TFRecordDataset(filenames_train)
    dataset_test_action = tf.data.TFRecordDataset(filenames_test_action)
    dataset_test_char = tf.data.TFRecordDataset(filenames_test_char)

    # example proto decode
    def _parse_function(example_proto):
        keys_to_features = {'frame': tf.FixedLenFeature((64, 64, 3), tf.float32),
                            'character_ID': tf.FixedLenFeature((), tf.int64, default_value=0),
                            'action_ID': tf.FixedLenFeature((), tf.int64, default_value=0)}
        parsed_features = tf.parse_single_example(example_proto, keys_to_features)
        return parsed_features['frame'], parsed_features['character_ID'], parsed_features['action_ID']

    dataset_train = dataset_train.map(_parse_function)
    dataset_test_action = dataset_test_action.map(_parse_function)
    dataset_test_char = dataset_test_char.map(_parse_function)

    dataset_train = dataset_train.batch(batch_size)
    dataset_test_action = dataset_test_action.batch(batch_size)
    dataset_test_char = dataset_test_char.batch(batch_size_test_char)

    # if shuffle_train:
    #     dataset_train = dataset_train.shuffle(buffer_size=batch_size)

    iterator = tf.data.Iterator.from_structure(dataset_train.output_types, dataset_train.output_shapes)
    training_init_op = iterator.make_initializer(dataset_train)
    test_action_init_op = iterator.make_initializer(dataset_test_action)
    test_char_init_op = iterator.make_initializer(dataset_test_char)

    return iterator, training_init_op, test_action_init_op, test_char_init_op


def sprites_PCA_init(path_train_dict, m=15, L_action=6, L_character=16, seed=42, N_action=72):
    """
    Produces PCA initialization for GPLVM action vectors as well as for inducing points for SPRITES dataset.

    :param path_train_dict:
    :param m: number of inducing point per action. Total number of inducing points = m*72
    :param L_action: dimension of GPLVM action vectors
    :param L_character: dimension of character vectors (needs to coincide with the dimension of
                                                                the output layer in representation network).
    :param seed:
    :param N_action:

    :return: GPLVM action init (N_action, L_action),
             inducing point init (N_action*m, L_action + L_character)

    """

    # read in entire training data
    train_dict = pickle.load(open(path_train_dict, 'rb'))
    train_frames, train_aux_data = train_dict['frames'], train_dict['aux_data']
    del train_dict

    # GPLVM action vectors: first compute average frame for each action, resulting in (72, 64*64*3) dataframe.
    # Then do PCA on it and keep only first L_action principal components, yielding (72, L_action) dataframe.
    action_ids_train = {i: [] for i in range(N_action)}
    for i in range(len(train_aux_data)):
        action_ids_train[train_aux_data[i, 1]].append(i)

    GPLVM_action = []
    for _, ids in action_ids_train.items():
        mean_frame = train_frames[ids].mean(axis=0).reshape(-1)
        GPLVM_action.append(mean_frame)

    pca_GPLVM = PCA(n_components=L_action)
    GPLVM_action = pca_GPLVM.fit_transform(np.array(GPLVM_action))

    # inducing points: for each GPLVM_action vector, sample m characters vectors from
    # PCA dataframe on the entire dataset (N_train, L_character)
    pca_global = PCA(n_components=L_character)
    train_frames_PCA = pca_global.fit_transform(train_frames.reshape(-1, 64 * 64 * 3))

    inducing_points = []
    for i in range(len(GPLVM_action)):
        char_vectors = []
        for pca_ax in range(L_character):
            # sample from empirical distribution of a given PCA feature
            pca_sample = scipy.stats.gaussian_kde(train_frames_PCA[:, pca_ax]).resample(m, seed=seed).reshape(-1)
            char_vectors.append(pca_sample)
        char_vectors = np.array(char_vectors).T

        # repeat GPLVM_action vector m times
        GPLVM_action_vec = GPLVM_action[i, :]
        GPLVM_action_vec = np.tile(GPLVM_action_vec, (m, 1))

        # hstack GPLVM action vector and character vectors
        inducing_points.append(np.hstack((GPLVM_action_vec, char_vectors)))

    inducing_points = np.concatenate(inducing_points)

    del train_frames
    del train_aux_data

    return GPLVM_action, inducing_points


def plot_sprites(arr, recon_arr, title, nr_images=8, seed=0):
    """

    :param arr:
    :param recon_arr:
    :param title:
    :param nr_images:
    :param seed:
    :return:
    """

    assert nr_images % 8 == 0

    if seed is not None:
        random.seed(seed)
        indices = random.sample(list(range(len(arr))), nr_images)
    else:
        indices = list(range(nr_images))
    plt.figure(figsize=(10, 10*int(nr_images/8)))
    plt.suptitle(title)
    for i in range(int(nr_images*2)):
        plt.subplot(int(nr_images / 2), 4, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        if i % 2 == 0:
            plt.imshow(arr[indices[i // 2]])
            plt.title("ID: {}".format(indices[i // 2]))
        else:
            plt.imshow(recon_arr[indices[i // 2]])
            plt.title("ID: {}".format(indices[i // 2]))
    # plt.tight_layout()
    plt.draw()


def aux_data_sprites_utils(batch_size, N, repeats):
    """
    Generates auxiliary arrays that are needed for construction of auxiliary data for SPRITES dataset.
    Generated arrays are later used in function aux_data_SVGPVAE_sprites.

    :param batch_size:
    :param N: nr. frames per character
    :param repeats: how many times is each summed character vector copied

    :return:
    """
    N_char = int(batch_size / N)  # nr. of unique characters
    segment_ids = np.array([[i] * N for i in range(N_char)]).reshape(-1)
    repeats_arr = [repeats for _ in range(N_char)]

    return segment_ids, repeats_arr


def forward_pass_pretraining_repr_NN(frames, labels, repr_NN, classification_layer, test_pipeline=False):
    """

    :param frames:
    :param labels:
    :param repr_NN: representation neural net
    :param classification_layer: classificiation layer, used only in the pretraining step
    :param test_pipeline:
    :return:
    """

    if not test_pipeline:
        # shuffle data in the batch. .shuffle did not work in import_sprites function...
        indices = tf.range(start=0, limit=tf.shape(frames)[0], dtype=tf.int32)
        shuffled_indices = tf.random.shuffle(indices)
        frames = tf.gather(frames, shuffled_indices)
        labels = tf.gather(labels, shuffled_indices)

    embeddings = repr_NN.repr_nn(frames)

    logits = classification_layer(embeddings)

    # labels_ = tf.one_hot(labels, 1000)

    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))

    if test_pipeline:
        preds = tf.argmax(logits, 1)
        _, acc = tf.compat.v1.metrics.accuracy(labels=labels, predictions=preds)

        return loss, acc

    else:
        return loss


def save_sprites(save_path, N=2000):
    """
    Saves preprocessed SPRITES datasets to .tfrecord files

    :param save_path:
    :param N: number of images per .tfrecord file

    """

    # create folders where we the SPRITES dataset will be saved
    Path(save_path + "/train").mkdir(parents=True, exist_ok=True)
    Path(save_path + "/test_action").mkdir(parents=True, exist_ok=True)
    Path(save_path + "/test_character").mkdir(parents=True, exist_ok=True)

    # save train dataset
    N_files = int(np.ceil(len(train_frames) / N))
    for i in range(N_files):
        filename = save_path + "/train/train{}.tfrecord".format(i + 1)
        npy_to_tfrecords(train_frames[i * N:(i + 1) * N], train_aux_data[i * N:(i + 1) * N], filename)
        print(filename + " written!")

    # save test_action dataset
    N_files = int(np.ceil(len(test_action_frames) / N))
    for i in range(N_files):
        filename = save_path + "/test_action/test_action{}.tfrecord".format(i + 1)
        npy_to_tfrecords(test_action_frames[i * N:(i + 1) * N], test_action_aux_data[i * N:(i + 1) * N], filename)
        print(filename + " written!")

    # save test_character dataset
    N_files = int(np.ceil(len(test_char_frames) / N))
    for i in range(N_files):
        filename = save_path + "/test_character/test_character{}.tfrecord".format(i + 1)
        npy_to_tfrecords(test_char_frames[i * N:(i + 1) * N], test_char_aux_data[i * N:(i + 1) * N], filename)
        print(filename + " written!")

    # save train frames and aux data so that can later use in PCA_init functions
    train_dict = {'frames': train_frames, 'aux_data': train_aux_data}
    pickle.dump(train_dict, open(save_path + '/sprites_train_dict.p', 'wb'))


if __name__ == "__main__":

    train_frames, train_aux_data, test_action_frames, \
    test_action_aux_data, test_char_frames, test_char_aux_data = preprocess_sprite_SVGPVAE(path=SPRITES_repo_path,
                                                                                           N_frames_train=50)

    save_sprites("SPRITES_data")