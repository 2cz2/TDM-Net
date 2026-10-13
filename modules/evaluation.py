import json
from pathlib import Path
from .contract import PROTOCOL, HISTORY_CONDITIONS, MECHANISM_CONDITIONS, json_hash
from .clinical_metrics import CHEXBERT_LABELS, binary

def validate_rows(records, labeled=False):
    rows, common = {}, None
    for r in records:
        key = (str(r['subject_id']), str(r['study_id']))
        if key in rows:
            raise ValueError('Duplicate ID: '+str(key))
        contract, action = r.get('generation_contract', {}), r.get('intervention', {})
        if r.get('protocol') != PROTOCOL or contract.get('protocol') != PROTOCOL:
            raise ValueError('Expected revision v2 generation provenance')
        for field in ('checkpoint_sha256','generation_code_sha256','annotation_sha256',
                      'metadata_sha256','training_settings_sha256'):
            if not isinstance(contract.get(field), str) or len(contract[field]) != 64:
                raise ValueError('Missing generation fingerprint: '+field)
        if not contract.get('decoding') or len(r.get('current_input_sha256','')) != 64:
            raise ValueError('Missing decoding/current input provenance')
        if r.get('generation_checkpoint_sha256') != contract['checkpoint_sha256']:
            raise ValueError('Checkpoint fingerprints disagree')
        history, mechanism = action.get('history'), action.get('mechanism')
        if history not in HISTORY_CONDITIONS or mechanism not in MECHANISM_CONDITIONS:
            raise ValueError('Invalid intervention provenance')
        if history != 'factual' and mechanism != 'none':
            raise ValueError('Combined interventions have no declared contrast')
        if r.get('history_intervention') != history or r.get('mechanism_intervention') != mechanism:
            raise ValueError('Intervention fields disagree')
        if not all(isinstance(r.get(f),str) for f in ('reference','prediction')):
            raise ValueError('Missing reference/prediction text')
        prior = r.get('previous_study_id') is not None
        if prior != (r.get('previous_report') is not None):
            raise ValueError('Prior ID/report availability differs')
        signature = [contract, action, r.get('experiment'), r.get('seed')]
        if labeled:
            provenance = r.get('labeler_contract')
            if (r.get('label_names') != CHEXBERT_LABELS or not provenance
                    or r.get('labeler_sha256') != json_hash(provenance)):
                raise ValueError('Missing or inconsistent labeler provenance/order')
            binary([r['reference_classes']]); binary([r['prediction_classes']])
            if prior:
                binary([r['previous_classes']])
            elif r.get('previous_classes') is not None:
                raise ValueError('First visits must have null previous labels')
            signature.append(provenance)
        if common is None:
            common = signature
        elif common != signature:
            raise ValueError('Mixed model, configuration, intervention or labeler in one file')
        rows[key] = r
    if not rows:
        raise ValueError('Empty evaluation file')
    return rows

def read(path, labeled=True):
    return validate_rows([json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()], labeled)

def aligned(left, right, counterfactual=False):
    if left.keys() != right.keys():
        raise ValueError('Study ID sets differ; no silent intersection/drop is permitted')
    for key in left:
        a, b = left[key], right[key]
        for field in ('reference','previous_report','previous_study_id','history_study_ids',
                      'current_image_paths','study_time','current_input_sha256','reference_classes',
                      'previous_classes','labeler_sha256','label_names'):
            if a.get(field) != b.get(field):
                raise ValueError('Paired data mismatch: {} / {}'.format(key,field))
        ca, cb = a['generation_contract'], b['generation_contract']
        for field in ('annotation_sha256','metadata_sha256','decoding'):
            if ca[field] != cb[field]:
                raise ValueError('Paired generation mismatch: '+field)
        if a['intervention'] != {'history':'factual','mechanism':'none'}:
            raise ValueError('The reference condition must be factual without mechanism intervention')
        if counterfactual:
            if ca != cb:
                raise ValueError('Counterfactual comparison requires identical checkpoint, code and settings')
            if a['intervention'] == b['intervention']:
                raise ValueError('Counterfactual condition must intervene on history or a mechanism')
        elif b['intervention'] != a['intervention']:
            raise ValueError('Cross-model baseline must also use factual inputs')
