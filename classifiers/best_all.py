from classifiers.estimators_all import CLASSIFIERS


def get_best_fitted(X_train, y_train):
    return {
        "rf-grid": CLASSIFIERS['rf-grid'].fit(X_train, y_train).best_estimator_,
        "svm-grid": CLASSIFIERS['svm-grid'].fit(X_train, y_train).best_estimator_,
        "nb-bernoulli-grid": CLASSIFIERS['nb-bernoulli-grid'].fit(X_train, y_train).best_estimator_,
        "knn-grid": CLASSIFIERS['knn-grid'].fit(X_train, y_train).best_estimator_
    }
