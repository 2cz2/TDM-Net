import argparse
import csv
import gzip
import json
from pathlib import Path
from modules.protocol import CLASS_NAMES, audit_annotation, identifier, training_labels, sha256
from modules.contract import atomic_json

def attach_labels(annotation, labels_path):
    ann=dict(annotation,train=[dict(r) for r in annotation['train']])
    before=audit_annotation(ann)
    labels,study_owners={},{}
    opener=gzip.open if str(labels_path).lower().endswith('.gz') else open
    with opener(labels_path,'rt',newline='',encoding='utf-8-sig') as f:
        reader=csv.DictReader(f)
        if not set(CLASS_NAMES+['subject_id','study_id']).issubset(reader.fieldnames or []):
            raise ValueError('CheXpert CSV needs subject_id, study_id and all 14 named label columns')
        for r in reader:
            key=(identifier(r['subject_id'],'p'),identifier(r['study_id'],'s'))
            values={c:r[c] for c in CLASS_NAMES}
            training_labels({'label':values},require=True)
            if key[1] in study_owners and study_owners[key[1]]!=key[0]:
                raise ValueError('Label CSV assigns study to multiple patients: '+key[1])
            study_owners[key[1]]=key[0]
            if key in labels and labels[key]!=values:
                raise ValueError('Conflicting supervision labels: '+str(key))
            labels[key]=values
    matched,supervised=0,0
    missing_examples=[]
    known_per_class=[0]*14
    for r in ann['train']:
        key=(identifier(r['subject_id'],'p'),identifier(r['study_id'],'s'))
        if key[1] in study_owners and study_owners[key[1]]!=key[0]:
            raise ValueError('Annotation/label CSV patient mismatch: '+str(key))
        found=key in labels
        matched+=int(found)
        r['label']=dict(labels[key]) if found else dict.fromkeys(CLASS_NAMES,None)
        r['label_names']=CLASS_NAMES
        r['label_source_status']='matched_chexpert_csv' if found else 'absent_from_chexpert_csv'
        _,mask=training_labels(r,require=True)
        supervised+=int(any(mask))
        known_per_class=[n+int(m) for n,m in zip(known_per_class,mask)]
        if not found and len(missing_examples)<20:
            missing_examples.append({'subject_id':key[0],'study_id':key[1]})
    if not supervised:
        raise ValueError('No known disease supervision could be connected to train. '
                         'Check IDs and the label file; no output annotation was saved.')
    if audit_annotation(ann)!=before: raise RuntimeError('Dataset membership changed')
    summary={'labels_csv_sha256':sha256(labels_path),'label_encoding':'CheXpert -1/0/1/blank',
             'training_studies':len(ann['train']),'matched_csv_studies':matched,
             'absent_from_csv':len(ann['train'])-matched,'supervised_studies':supervised,
             'studies_without_known_labels':len(ann['train'])-supervised,
             'known_labels_per_class':dict(zip(CLASS_NAMES,known_per_class)),
             'missing_examples':missing_examples,
             'unknown_policy':'retain_report_loss_mask_auxiliary_classification'}
    ann['training_label_provenance']=summary
    return ann,summary


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--annotation',required=True)
    p.add_argument('--labels_csv',required=True,help='Official CheXpert CSV or CSV.gz')
    p.add_argument('--output',required=True)
    a=p.parse_args()
    target=Path(a.output)
    if target.exists(): raise FileExistsError('Use a new annotation path')
    annotation=json.loads(Path(a.annotation).read_text(encoding='utf-8'))
    ann,summary=attach_labels(annotation,a.labels_csv)
    summary['input_annotation_sha256']=sha256(a.annotation)
    atomic_json(target,ann)
    print(json.dumps(summary,indent=2))
    print('Saved new annotation with training-only disease supervision. Original untouched.')
