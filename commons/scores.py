def precision_recall(kw_true, kw_pred):
    # we can use sets since words (i.e., graph nodes) are all uniques
    tp = set(kw_true) & set(kw_pred)
    fp = set(kw_pred) - set(kw_true)
    fn = set(kw_true) - set(kw_pred)

    # P = TP / (TP + FP)
    precision = len(tp) / (len(tp) + len(fp)) if len(tp) + len(fp) > 0 else 0
    # R = TP / (TP + FN)
    recall = len(tp) / (len(tp) + len(fn)) if len(tp) + len(fn) > 0 else 0

    return precision, recall
