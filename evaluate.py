import argparse
import json
from pathlib import Path
import numpy as np
from modules.clinical_metrics import CHEXBERT_LABELS, ce_metrics, inertia, paired_patient_bootstrap

from modules.evaluation import read, aligned
from modules.contract import atomic_json
from modules.protocol import sha256

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--factual', required=True)
    p.add_argument('--counterfactual')
    p.add_argument('--baseline', help='A different trained model on identical target IDs')
    p.add_argument('--output', required=True)
    p.add_argument('--bootstrap', type=int, default=1000)
    p.add_argument('--seed', type=int, default=9233)
    a = p.parse_args()
    if a.bootstrap < 2:
        p.error('--bootstrap must be at least 2')
    rows = read(a.factual)
    if next(iter(rows.values()))['intervention'] != {'history':'factual','mechanism':'none'}:
        raise ValueError('Primary input must be factual')
    keys = sorted(rows)
    gt = np.asarray([rows[k]['reference_classes'] for k in keys])
    pr = np.asarray([rows[k]['prediction_classes'] for k in keys])
    prev = np.asarray([rows[k]['previous_classes'] or [np.nan]*14 for k in keys])
    cf = None
    if a.counterfactual:
        other = read(a.counterfactual); aligned(rows,other,counterfactual=True)
        cf = np.asarray([other[k]['prediction_classes'] for k in keys])
    # History-removal CF-HI and fixed-mechanism contrasts are separate estimands.
    mechanism = cf is not None and other[keys[0]]['intervention']['mechanism'] != 'none'
    history_cf = None if mechanism else cf
    result = {'ce': ce_metrics(gt,pr), 'ce_uncertain_positive': ce_metrics(gt,pr,True),
              'hi': inertia(gt,pr,prev,history_cf), 'hi_blank_negative': inertia(gt,pr,prev,history_cf,True)}
    subjects = [k[0] for k in keys]
    if cf is not None:
        result['intervention'] = other[keys[0]]['intervention']
        if mechanism:
            contrast = inertia(gt,pr,prev,cf)
            result['mechanism_transition_effects'] = {
                'factual_FN_error_removed':contrast['CF_HI_FN'],
                'factual_FP_error_removed':contrast['CF_HI_FP'],
                'FN_error_introduced':contrast['history_helpful_FN'],
                'FP_error_introduced':contrast['history_helpful_FP'],
                'net_FN_error_reduction_rate':contrast['net_history_FN_rate'],
                'net_FP_error_reduction_rate':contrast['net_history_FP_rate']}
            metrics = ('net_history_FN_rate','net_history_FP_rate')
        else:
            metrics = ('CF_HI_FN_rate','CF_HI_FP_rate','net_history_FN_rate','net_history_FP_rate')
        for metric in metrics:
            name = metric if not mechanism else metric.replace('net_history_', 'mechanism_net_')
            result[name+'_bootstrap'] = paired_patient_bootstrap(subjects,
                lambda ix: inertia(gt[ix],pr[ix],prev[ix],cf[ix])[metric], a.bootstrap,a.seed)
    if a.baseline:
        base = read(a.baseline); aligned(rows,base)
        bp = np.asarray([base[k]['prediction_classes'] for k in keys])
        for metric in ('micro_f1_14','macro_f1_14'):
            result['delta_'+metric] = ce_metrics(gt,pr)[metric] - ce_metrics(gt,bp)[metric]
            result['delta_'+metric+'_bootstrap'] = paired_patient_bootstrap(subjects,
                lambda ix: ce_metrics(gt[ix],pr[ix])[metric] - ce_metrics(gt[ix],bp[ix])[metric],a.bootstrap,a.seed)
    result['input_sha256'] = {name:sha256(path) for name,path in
        [('factual',a.factual),('counterfactual',a.counterfactual),('baseline',a.baseline)] if path}
    result['generation_contract'] = rows[keys[0]]['generation_contract']
    result['labeler_contract'] = rows[keys[0]]['labeler_contract']
    atomic_json(a.output,result)
    print(json.dumps(result,indent=2,allow_nan=False))
