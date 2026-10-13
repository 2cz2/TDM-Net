"""Recompute full CE from existing CheXbert labels; no neural inference or training."""
import argparse
import json
from pathlib import Path
import numpy as np
from modules.clinical_metrics import binary, ce_metrics, inertia, CHEXBERT_LABELS
from modules.evaluation import read
from modules.contract import atomic_json
from modules.protocol import sha256


def complete_metrics(gt_classes, pred_classes, uncertain_positive=False):
    result = ce_metrics(gt_classes, pred_classes, uncertain_positive)
    gt, pr = binary(gt_classes, uncertain_positive), binary(pred_classes, uncertain_positive)
    tp, fp, fn = (gt & pr).sum(0), (~gt & pr).sum(0), (gt & ~pr).sum(0)
    divide = lambda n, d: np.divide(n, d, out=np.zeros_like(n, dtype=float), where=d > 0)
    p, r = divide(tp, tp + fp), divide(tp, tp + fn)
    result.update(accuracy_element=float((gt == pr).mean()),
                  accuracy_exact=float((gt == pr).all(1).mean()),
                  micro_precision=float(tp.sum() / (tp + fp).sum()) if (tp + fp).sum() else 0.,
                  micro_recall=float(tp.sum() / (tp + fn).sum()) if (tp + fn).sum() else 0.,
                  macro_precision=float(p.mean()), macro_recall=float(r.mean()),
                  per_class_precision=dict(zip(CHEXBERT_LABELS, p.tolist())),
                  per_class_recall=dict(zip(CHEXBERT_LABELS, r.tolist())),
                  per_class_counts={name: {'tp': int(tp[i]), 'fp': int(fp[i]), 'fn': int(fn[i]),
                                          'tn': int((~gt & ~pr).sum(0)[i])}
                                    for i, name in enumerate(CHEXBERT_LABELS)})
    return result


def evaluate(path):
    records = read(path)
    rows = list(records.values())
    splits = {r.get('evaluation_split') for r in rows}
    if len(splits) != 1 or next(iter(splits)) not in ('val', 'test'):
        raise ValueError('Mixed or missing evaluation split')
    gt, pr = [r['reference_classes'] for r in rows], [r['prediction_classes'] for r in rows]
    prev = [r['previous_classes'] or [float('nan')] * 14 for r in rows]
    return dict(split=next(iter(splits)), n_reports=len(rows),
                n_subjects=len({k[0] for k in records}),
                ce=complete_metrics(gt, pr), ce_uncertain_positive=complete_metrics(gt, pr, True),
                hi=inertia(gt, pr, prev), hi_blank_negative=inertia(gt, pr, prev, blank_negative=True),
                generation_contract=rows[0]['generation_contract'],
                labeler_contract=rows[0]['labeler_contract'], input_sha256=sha256(path))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    result = evaluate(a.input)
    atomic_json(a.output, result)
    keys = ('accuracy_element', 'accuracy_exact', 'micro_precision', 'micro_recall',
            'micro_f1_14', 'macro_precision', 'macro_recall', 'macro_f1_14')
    print(json.dumps({'split': result['split'], 'n_reports': result['n_reports'],
                      **{k: result['ce'][k] for k in keys}}, indent=2), flush=True)


if __name__ == '__main__':
    main()
