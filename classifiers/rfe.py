from sklearn import feature_selection
from classifiers.estimators_all import CLASSIFIERS
from classifiers.best_all import get_best_fitted

def get_rfe(X_train, y_train, step=5):
    BEST_FITTED = get_best_fitted(X_train, y_train)
    RFE = {
        "svm-rfe": {
            "model": feature_selection.RFE(CLASSIFIERS['svm'], step=step),
            "estimator": CLASSIFIERS['svm']
        },
        "rf-rfe": {
            "model": feature_selection.RFE(CLASSIFIERS['rf'], step=step),
            "estimator": CLASSIFIERS['rf']
        },
        "knn-rfe": {
            "model": feature_selection.RFE(CLASSIFIERS['rf'], step=step),
            "estimator": CLASSIFIERS['knn']
        },
        "nb-bernoulli-rfe": {
            "model": feature_selection.RFE(CLASSIFIERS['nb-bernoulli'], step=step),
            "estimator": CLASSIFIERS['nb-bernoulli']
        },
        "svm-grid-rfe": {
            "model": feature_selection.RFE(BEST_FITTED['svm-grid'], step=step),
            "estimator": BEST_FITTED['svm-grid']
        },
        "rf-grid-rfe": {
            "model": feature_selection.RFE(BEST_FITTED['rf-grid'], step=step),
            "estimator": BEST_FITTED['rf-grid']
        },
        "knn-grid-rfe": {
            "model": feature_selection.RFE(BEST_FITTED['rf-grid'], step=step),
            "estimator": BEST_FITTED['knn-grid']
        },
    }

    return RFE
