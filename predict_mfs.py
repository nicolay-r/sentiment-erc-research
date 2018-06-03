#!/usr/bin/python
# -*- coding: utf-8 -*-

from sklearn import feature_selection

from core.classification import evaluate, fit_and_predict, filter_features, \
    create_train_data, create_test_data, create_model_feature_importances, \
    plot_feature_importances

import io_utils
from classifiers.estimators_all import CLASSIFIERS
from classifiers.best_all import get_best_fitted

#
# MAIN
#
synonyms_filepath = io_utils.get_synonyms_filepath()

print "Preparing Train Collection"
X_train, y_train = create_train_data(io_utils.get_train_vectors_list())
print "Preparing Test Collection"
X_test, test_collections = create_test_data(io_utils.get_test_vectors_list())

BEST_FITTED = get_best_fitted(X_train, y_train)

FSM = {
    "rf-fs": {
        "model": feature_selection.SelectFromModel(CLASSIFIERS['rf'], prefit=False),
        "estimator": CLASSIFIERS['rf']
    },
    "svm-fs": {
        "model": feature_selection.SelectFromModel(CLASSIFIERS['svm'], prefit=False),
        "estimator": CLASSIFIERS['svm']
    },
    "knn-fs": {
        "model": feature_selection.SelectFromModel(CLASSIFIERS['rf'], prefit=False),
        "estimator": CLASSIFIERS['knn']
    },
    "nb-bernoulli-fs": {
        "model": feature_selection.SelectFromModel(CLASSIFIERS['nb-bernoulli'], prefit=False),
        "estimator": CLASSIFIERS['nb-bernoulli']
    },
    "rf-grid-fs": {
        "model": feature_selection.SelectFromModel(BEST_FITTED['rf-grid'], prefit=True),
        "estimator": BEST_FITTED['rf-grid']
    },
    "svm-grid-fs": {
        "model": feature_selection.SelectFromModel(BEST_FITTED['svm-grid'], prefit=True),
        "estimator": BEST_FITTED['svm-grid']
    },
    "nb-bernoulli-grid-fs": {
        "model": feature_selection.SelectFromModel(BEST_FITTED['nb-bernoulli-grid'], prefit=True),
        "estimator": BEST_FITTED['nb-bernoulli-grid']
    },
    "knn-grid-fs": {
        "model": feature_selection.SelectFromModel(BEST_FITTED['rf-grid'], prefit=True),
        "estimator": BEST_FITTED['knn-grid']
    },
}

# features selection
for model_name in FSM:
    model = FSM[model_name]['model']
    estimator = FSM[model_name]['estimator']
    if not model.prefit:
        model.fit(X_train, y_train)
    idf = create_model_feature_importances(
            model.estimator,
            model_name,
            model.get_support(),
            io_utils.read_feature_names())
    print "Feature selection for SFM method: {}".format(model_name)
    X_train_new, X_test_new = filter_features(model_name, model, X_train, X_test)
    test_opinions = fit_and_predict(model_name,
                                    estimator,
                                    X_train_new,
                                    y_train,
                                    X_test_new,
                                    test_collections,
                                    synonyms_filepath)
    io_utils.save_test_opinions(test_opinions, model_name)

    edf = evaluate(FSM[model_name]["estimator"],
                   model_name,
                   io_utils.create_files_to_compare_list(model_name),
                   io_utils.get_method_root(model_name),
                   synonyms_filepath)

    e_filepath = "{}/test_{}.csv".format(io_utils.eval_sfm_root(), model_name)
    i_filepath = "{}/test_{}.features.csv".format(io_utils.eval_features_filepath(), model_name)
    fig_filepath = '{}/test_{}.png'.format(io_utils.eval_root(), model_name)

    edf.to_csv(e_filepath)
    idf.to_csv(i_filepath)
    fig = plot_feature_importances(i_filepath)
    fig.savefig(fig_filepath)
