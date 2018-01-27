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

    fmt = f"""
    Area under ROC curve:                                        {auc:.2f}%
    True negatives  (correctly predicted normal use):   {tn:6d} ({tn_prct:.2f}%)
    False positives (incorrectly predicted attacks):    {fp:6d} ({fp_prct:.2f}%)
    False negatives (incorrectly predicted normal use): {fn:6d} ({fn_prct:.2f}%)
    True positives  (correctly predicted attacks):      {tp:6d} ({tp_prct:.2f}%)
    """
    print(fmt)

    return results
