#!/usr/bin/python
import io
from os import path, makedirs, mkdir
from core.statistic import FilesToCompare

ignored_entity_values = [u"author", u"unknown"]


def get_server_root():
    return '/home/nicolay/storage/disk/homes/nicolay/datasets/news'

def get_w2v_model_filepath():
    return 'data/w2v_model.bin.gz'

def read_prepositions(filepath):
    prepositions = []
    with io.open(filepath, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            prepositions.append(line.strip())

    return prepositions


def read_lss(filepath):
    words = []
    with io.open(filepath, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            words.append(line.lower().strip())

    return words


def read_feature_names():
    features = []
    with open(get_feature_names_filepath(), 'r') as f:
        for l in f.readlines():
            features.append(l.strip())
    return features


def train_indices():
    indices = range(1, 46)
    for i in [9, 22, 26]:
        if i in indices:
            indices.remove(i)
    return indices


def test_indices():
    indices = range(46, 76)
    for i in [70]:
        if i in indices:
            indices.remove(i)
    return indices


def get_ignored_entity_values():
    return ignored_entity_values


def data_root():
    return "data/"


def test_root():
    return path.join(data_root(), "test/")


def train_root():
    return path.join(data_root(), "train/")


def eval_root():
    return path.join(data_root(), "Eval/")


def eval_rfe_root():
    result = path.join(eval_root(), "rfe")
    if not path.exists(result):
        makedirs(result)
    return result


def eval_sfm_root():
    result = path.join(eval_root(), "sfm")
    if not path.exists(result):
        makedirs(result)
    return result


def eval_ec_root():
    """ class elemination root
    """
    result = path.join(eval_root(), "ce")
    if not path.exists(result):
        makedirs(result)
    return result


def eval_univariate_root():
    result = path.join(eval_root(), "uv")
    if not path.exists(result):
        makedirs(result)
    return result


def eval_default_root():
    result = path.join(eval_root(), "default")
    if not path.exists(result):
        makedirs(result)
    return result

def eval_baseline_root():
    result = path.join(eval_root(), "baseline")
    if not path.exists(result):
        makedirs(result)
    return result


def eval_features_filepath():
    return path.join(eval_root(), 'features')


def get_method_root(method_name):
    result = path.join(test_root(), method_name)
    if not path.exists(result):
        makedirs(result)
    return result


def get_train_vectors_list():
    return [path.join(train_root(), "art{}.vectors.txt".format(i)) for i in train_indices()]


def get_test_vectors_list():
    return [path.join(test_root(), "art{}.vectors.txt".format(i)) for i in test_indices()]


def get_etalon_root():
    return path.join(data_root(), "etalon/")


def get_synonyms_filepath():
    return path.join(data_root(), "synonyms.txt")


def get_feature_names_filepath():
    return path.join(data_root(), "feature_names.txt")


def save_test_opinions(test_opinions, method_name):
    """
    Save list of opinions
    """
    method_root = get_method_root(method_name)
    if not path.exists(method_root):
        mkdir(method_root)

    for i, test_index in enumerate(test_indices()):
        # TODO. should guarantee that order the same as during reading operation.
        opin_filepath = "{}/art{}.opin.txt".format(method_root, test_index)
        test_opinions[i].save(opin_filepath)


def create_files_to_compare_list(method_name):
    """
    Create list of comparable opinion files for the certain method.
    method_name: str
    """
    method_root_filepath = get_method_root(method_name)
    return [FilesToCompare(
                "{}/art{}.opin.txt".format(method_root_filepath, i),
                "{}/art{}.opin.txt".format(get_etalon_root(), i),
                i)
            for i in test_indices()]


