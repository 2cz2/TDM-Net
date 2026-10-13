"""Label raw full reference/prior/predicted reports while preserving IDs and label provenance."""
import argparse
import inspect
import json
from pathlib import Path
import torch
from torch import nn
from transformers import BertModel, BertTokenizer
import transformers
from modules.clinical_metrics import CHEXBERT_LABELS
from modules.protocol import sha256
from modules.contract import json_hash, atomic_text, implementation_hash
from modules.evaluation import validate_rows

class CheXbert(nn.Module):
    def __init__(self, bert_path):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_path, local_files_only=True)
        self.linear_heads = nn.ModuleList([nn.Linear(self.bert.config.hidden_size,4) for _ in range(13)]
                                        + [nn.Linear(self.bert.config.hidden_size,2)])
    def forward(self, **inputs):
        x = self.bert(**inputs).last_hidden_state[:,0]
        return torch.stack([head(x).argmax(dim=-1) for head in self.linear_heads],dim=1)


def load_chexbert_state(model, state):
    """Adapt only deterministic BERT ID buffers; strictly load all learned weights."""
    state = dict(state)
    expected = model.state_dict()
    buffers = dict(model.named_buffers())
    parameters = dict(model.named_parameters())
    changes = []
    for name in ('bert.embeddings.position_ids', 'bert.embeddings.token_type_ids'):
        if name in parameters:
            raise RuntimeError('Expected a non-learned BERT buffer: ' + name)
        current = buffers.get(name)
        saved = state.get(name)
        for value in (current, saved):
            if value is None:
                continue
            if (not torch.is_tensor(value) or value.dtype != torch.long or
                    value.ndim != 2 or value.shape[0] != 1 or value.shape[1] < 1):
                raise RuntimeError('Invalid deterministic BERT buffer: ' + name)
            canonical = (torch.arange(value.shape[1], device=value.device).unsqueeze(0)
                         if name.endswith('position_ids') else torch.zeros_like(value))
            if not torch.equal(value, canonical):
                raise RuntimeError('Noncanonical BERT buffer; refusing to ignore it: ' + name)
        if name in expected and name not in state:
            if current is None:
                raise RuntimeError('Missing initialized BERT buffer: ' + name)
            state[name] = current.detach().clone()
            changes.append('restored deterministic buffer: ' + name)
        elif name in state and name not in expected:
            # Some Transformers versions register these as non-persistent buffers.
            if current is None or current.shape != saved.shape:
                raise RuntimeError('Cannot verify checkpoint BERT buffer: ' + name)
            del state[name]
            changes.append('omitted verified non-persistent buffer: ' + name)
    model.load_state_dict(state, strict=True)
    return changes


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--chexbert', required=True)
    p.add_argument('--bert', required=True)
    p.add_argument('--batch_size', type=int, default=32)
    a = p.parse_args()
    target = Path(a.output)
    if target.exists():
        raise FileExistsError('Refusing to overwrite labeled reports')
    rows = [json.loads(line) for line in Path(a.input).read_text(encoding='utf-8').splitlines()]
    validate_rows(rows)
    if a.batch_size < 1:
        p.error('--batch_size must be positive')
    receipt_path = Path(a.input).with_name(Path(a.input).stem+'_complete.json')
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    if receipt.get('prediction_sha256') != sha256(a.input) or receipt.get('n_records') != len(rows):
        raise ValueError('Input predictions differ from their completed-test receipt')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CheXbert(a.bert)
    options = {'map_location':'cpu'}
    if 'weights_only' in inspect.signature(torch.load).parameters:
        options['weights_only'] = False
    state = torch.load(a.chexbert, **options)
    state = state.get('model_state_dict', state.get('state_dict',state))
    state = {k.removeprefix('module.'): v for k,v in state.items()}
    compatibility = load_chexbert_state(model, state)
    print('CheXbert strict parameter loading passed; buffer compatibility: {}'.format(
        compatibility or 'none required'), flush=True)
    model.to(device).eval()
    tokenizer = BertTokenizer.from_pretrained(a.bert, local_files_only=True)
    # Caching identical report labels is safe; records are NOT deduplicated or text-joined.
    unique = list(dict.fromkeys(r[field] for r in rows for field in ('reference','prediction','previous_report') if r.get(field) is not None))
    cache = {}
    with torch.no_grad():
        for start in range(0,len(unique),a.batch_size):
            reports = unique[start:start+a.batch_size]
            tokens = tokenizer(reports,return_tensors='pt',padding=True,truncation=True,max_length=512)
            labels = model(**{k:v.to(device) for k,v in tokens.items()}).cpu().tolist()
            cache.update(zip(reports,labels))
            print('Labeled {}/{} unique report texts'.format(
                min(start+a.batch_size, len(unique)), len(unique)), flush=True)
    provenance = {'checkpoint_sha256':sha256(a.chexbert),
                  'bert_config':model.bert.config.to_dict(),
                  'tokenizer_vocab_sha256':json_hash(tokenizer.get_vocab()),
                  'special_tokens':tokenizer.special_tokens_map,
                  'do_lower_case':tokenizer.do_lower_case,
                  'tokenizer_type':type(tokenizer).__name__,
                  'truncation_length':512,'padding_side':tokenizer.padding_side,
                  'truncation_side':tokenizer.truncation_side,
                  'transformers_version':transformers.__version__, 'torch_version':torch.__version__,
                  'implementation_sha256':implementation_hash(__file__), 'label_names':CHEXBERT_LABELS,
                  'checkpoint_buffer_compatibility':compatibility}
    # The local directory name is not a model property and may differ between servers.
    provenance['bert_config'].pop('_name_or_path',None)
    labeler_hash = json_hash(provenance)
    for r in rows:
        r['reference_classes'] = cache[r['reference']]
        r['prediction_classes'] = cache[r['prediction']]
        r['previous_classes'] = cache[r['previous_report']] if r.get('previous_report') is not None else None
        r['labeler_sha256'], r['label_names'] = labeler_hash, CHEXBERT_LABELS
        r['labeler_contract'] = provenance
    validate_rows(rows,labeled=True)
    atomic_text(target,''.join(json.dumps(r)+'\n' for r in rows))
    print('Labeled {} reports; label order and checkpoint fingerprint saved.'.format(len(rows)))
