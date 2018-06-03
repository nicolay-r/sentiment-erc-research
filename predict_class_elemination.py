#!/usr/bin/python
# -*- coding: utf-8 -*-


from os import path
import pandas as pd
import io_utils
from core.classification import filter_features_by_mask, evaluate, \
        create_test_data, create_train_data, fit_and_predict
from classifiers.estimators_all import CLASSIFIERS


def get_feature_groups(feature_list):
    """
    feature_list: list
        list of feature names

    return: dict
        dictionary of different feature groups
    """
    assert(type(feature_list) == list)
    groups = {}
    for feature_index, feature_name in enumerate(feature_list):
        feature_class = feature_name.split('_')[0]
        if feature_class not in groups:
            groups[feature_class] = [feature_index]
        else:
            groups[feature_class].append(feature_index)
    return groups


def set_result(df, group_index, feature_names, feature_mask, method_name, f1):
    assert(isinstance(df, pd.DataFrame))
    assert(type(feature_names) == list)
    assert(type(feature_mask) == list)
    assert(len(feature_mask) == len(feature_names))

    if group_index not in df.index:
        df.loc[group_index] = None
        for index, name in enumerate(feature_names):
            df[name][group_index] = '+' if feature_mask[index] else ' '

    if method_name not in df.columns:
        df[method_name] = pd.Series(index=df.index)

    df[method_name][group_index] = f1


def and_masks(mask_1, mask_2):
    assert(type(mask_1) == list)
    assert(type(mask_2) == list)
    assert(len(mask_1) == len(mask_2))
    return [mask_1[i] and mask_2[i] for i in range(len(mask_1))]


def exclude_group(excluded_ids, features_count):
    return [i not in excluded_ids for i in range(features_count)]


def is_excluded_group(excluded_ids, mask):
    for i in excluded_ids:
        if mask[i] is False:
            return True
    return False


#
# MAIN
#
synonyms_filepath = io_utils.get_synonyms_filepath()
X_train, y_train = create_train_data(io_utils.get_train_vectors_list())
X_test, test_collections = create_test_data(io_utils.get_test_vectors_list())
feature_names = io_utils.read_feature_names()
features_count = len(feature_names)
feature_groups = get_feature_groups(feature_names)
feature_groups_names = list(feature_groups.iterkeys())

# classfiers with predefined settings
df = pd.DataFrame(columns=[feature_names])

global_mask = [True] * features_count
for r_count in range(len(feature_groups)):

    f_avg = []
    for group_index, group_name in enumerate(feature_groups_names):

        excluded_ids = feature_groups[group_name]
        if is_excluded_group(excluded_ids, global_mask):
            continue

        mask = and_masks(global_mask, exclude_group(excluded_ids, features_count))
        X_train_new, X_test_new = filter_features_by_mask(X_train, X_test, mask)

        f_methods = []
        for method_name in CLASSIFIERS:

            # if 'grid' not in method_name:
            #     continue

            test_opinions = fit_and_predict(
                    method_name,
                    CLASSIFIERS[method_name],
                    X_train_new,
                    y_train,
                    X_test_new,
                    test_collections,
                    synonyms_filepath)

            io_utils.save_test_opinions(test_opinions, method_name)

            # edf = evaluate(CLASSIFIERS[method_name],
            #             method_name,
            #             io_utils.create_files_to_compare_list(method_name),
            #             io_utils.get_method_root(method_name),
            #             synonyms_filepath)

            # f1 = edf.loc['f1'][0]
            f1 = 1
            f_methods.append(f1)

            row_index = len(feature_groups) * r_count + group_index
            set_result(df, row_index, feature_names, mask, method_name, f1)
            df.to_csv(path.join(io_utils.eval_ec_root(), "results.csv"))

        f_avg.append(sum(f_methods) * 1.0 / len(f_methods))

    exclude_ind = f_avg.index(min(f_avg))
    excluded_ids = feature_groups[feature_groups_names[exclude_ind]]
    global_mask = and_masks(global_mask,
                            exclude_group(excluded_ids, features_count))
