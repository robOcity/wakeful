from sklearn.metrics import roc_auc_score, confusion_matrix

def print_scores(estimator_name=None, data_name=None, estimator=None, X_test=None, y_test=None):

    print('\n\n----- Estimator: {} ----- Data: {} ---------------------\n'.format(estimator_name, data_name))
    prob_predictions = estimator.predict_proba(X_test)
    auc = roc_auc_score(y_test, prob_predictions[:, 1])
    tn, fp, fn, tp = confusion_matrix(y_test, estimator.predict(X_test)).ravel()
    total = len(y_test)

    results = {'auc': auc,
               'tn': tn, 'tn_prct': 100*tn/total,
               'fp': fp, 'fp_prct': 100*fp/total,
               'fn': fn, 'fn_prct': 100*fn/total,
               'tp': tp, 'tp_prct': 100*tp/total,
               }

    print("""
    Area under ROC curve:                                      {:.2f}%
    True negatives  (correctly predicted normal use):   {:6d} ({:.2f}%)
    False positives (incorrectly predicted attacks):    {:6d} ({:.2f}%)
    False negatives (incorrectly predicted normal use): {:6d} ({:.2f}%)
    True positives  (correctly predicted attacks):      {:6d} ({:.2f}%)
    """.format(results))

    return results
