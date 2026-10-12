"""Report-label metrics and paired model-input interventions; no clinical causal claim."""
import numpy as np

CHEXBERT_LABELS = ['Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
                  'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
                  'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture',
                  'Support Devices', 'No Finding']

def binary(classes, uncertain_positive=False):
    a = np.asarray(classes)
    if a.ndim != 2 or a.shape[1] != 14 or not len(a):
        raise ValueError('Expected nonempty [N,14] CheXbert head classes')
    if not np.isin(a[:, :13], [0, 1, 2, 3]).all() or not np.isin(a[:, 13], [0, 1]).all():
        raise ValueError('Invalid CheXbert class values/order')
    y = a == 1
    if uncertain_positive:
        y[:, :13] |= a[:, :13] == 3
    return y

def ce_metrics(reference, prediction, uncertain_positive=False):
    gt, pr = binary(reference, uncertain_positive), binary(prediction, uncertain_positive)
    if gt.shape != pr.shape:
        raise ValueError('Reference/prediction shape mismatch')
    tp = (gt & pr).sum(axis=0)
    fp = (~gt & pr).sum(axis=0)
    fn = (gt & ~pr).sum(axis=0)
    den = 2*tp+fp+fn
    f1 = np.divide(2*tp, den, out=np.zeros(14, dtype=float), where=den>0)
    return {'micro_f1_14': float(2*tp.sum()/den.sum()) if den.sum() else 0.0,
            'macro_f1_14': float(f1.mean()), 'macro_f1_12_findings': float(f1[:12].mean()),
            'per_class_f1': dict(zip(CHEXBERT_LABELS, f1.tolist())),
            'per_class_support': dict(zip(CHEXBERT_LABELS, gt.sum(axis=0).tolist())),
            'n_reports': len(gt), 'zero_division': 0,
            'uncertain_policy': 'positive' if uncertain_positive else 'negative'}

def inertia(reference, prediction, previous, counterfactual=None, blank_negative=False):
    gt, pr, prev = np.asarray(reference), np.asarray(prediction), np.asarray(previous)
    binary(gt); binary(pr)
    if gt.shape != pr.shape or prev.shape != gt.shape:
        raise ValueError('HI input shape mismatch')
    # NaN prior rows denote first visits; they cannot enter either risk set.
    missing = np.isnan(prev).all(axis=1)
    if np.isnan(prev[~missing]).any():
        raise ValueError('Previous labels must be a complete row or an absent prior')
    if (~missing).any():
        binary(prev[~missing])
    cols = slice(0,12)  # No Finding and Support Devices are excluded from primary HI.
    negative = lambda a: (a == 2) | ((a == 0) & blank_negative)
    new = negative(prev[:,cols]) & (gt[:,cols] == 1)
    resolved = (prev[:,cols] == 1) & negative(gt[:,cols])
    missed = pr[:,cols] != 1
    propagated = pr[:,cols] == 1
    rate = lambda num, den: float(num/den) if den else None
    n_new, n_resolved = int(new.sum()), int(resolved.sum())
    fn, fp = int((new & missed).sum()), int((resolved & propagated).sum())
    result = {'HI_FN': fn, 'HI_FP': fp, 'new_opportunities': n_new,
              'resolved_opportunities': n_resolved, 'HI_FN_rate': rate(fn,n_new),
              'HI_FP_rate': rate(fp,n_resolved), 'n_targets': len(gt),
              'n_with_prior': int(np.isfinite(prev).all(axis=1).sum()),
              'n_targets_with_eligible_transition': int((new|resolved).any(axis=1).sum()),
              'label_set': CHEXBERT_LABELS[:12],
              'negative_policy': 'blank_or_negative' if blank_negative else 'explicit_negative_only'}
    if counterfactual is not None:
        cf = np.asarray(counterfactual)
        binary(cf)
        if cf.shape != gt.shape:
            raise ValueError('Counterfactual shape mismatch')
        harmful_fn = int((new & missed & (cf[:,cols] == 1)).sum())
        helpful_fn = int((new & ~missed & (cf[:,cols] != 1)).sum())
        harmful_fp = int((resolved & propagated & (cf[:,cols] != 1)).sum())
        helpful_fp = int((resolved & ~propagated & (cf[:,cols] == 1)).sum())
        result.update(CF_HI_FN=harmful_fn, CF_HI_FP=harmful_fp,
                      CF_HI_FN_rate=rate(harmful_fn,n_new), CF_HI_FP_rate=rate(harmful_fp,n_resolved),
                      history_helpful_FN=helpful_fn, history_helpful_FP=helpful_fp,
                      net_history_FN_rate=rate(harmful_fn-helpful_fn,n_new),
                      net_history_FP_rate=rate(harmful_fp-helpful_fp,n_resolved))
    return result

def paired_patient_bootstrap(subjects, statistic, iterations=1000, seed=9233):
    """Statistic accepts sampled row indices. Duplicate sampled patients retain all visits."""
    subjects = np.asarray(subjects)
    unique = np.unique(subjects)
    groups = {s: np.flatnonzero(subjects == s) for s in unique}
    if len(unique) < 2 or iterations < 2:
        raise ValueError('Bootstrap requires >=2 patients and >=2 iterations')
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(iterations):
        ix = np.concatenate([groups[s] for s in rng.choice(unique, len(unique), replace=True)])
        value = statistic(ix)
        if value is not None and np.isfinite(value):
            estimates.append(value)
    return {'ci95': np.percentile(estimates, [2.5,97.5]).tolist() if estimates else None,
            'valid_resamples': len(estimates), 'requested_resamples': iterations,
            'resampling_unit': 'subject', 'seed': seed}
