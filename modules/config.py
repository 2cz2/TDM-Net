"""Predeclared Table 3 experiments; each switch has one explicit meaning."""
EXPERIMENTS = {
    # history images, history text, CMI, TE, MG, DGL
    'current_only': (0, 0, 0, 0, 0, 0),
    'current_dgl': (0, 0, 0, 0, 0, 1),
    'current_mg_dgl': (0, 0, 0, 0, 1, 1),
    'direct_concat': (1, 1, 0, 0, 0, 0),
    'longitudinal': (1, 1, 1, 1, 0, 0),
    'longitudinal_mg': (1, 1, 1, 1, 1, 0),
    'longitudinal_dgl': (1, 1, 1, 1, 0, 1),
    'full': (1, 1, 1, 1, 1, 1),
    'history_image_only': (1, 0, 1, 1, 1, 1),
    'history_report_only': (0, 1, 1, 1, 1, 1),
}

def configure(args):
    names = ('history_images', 'history_reports', 'use_cmi', 'use_te', 'use_mg', 'use_dgl')
    for name, value in zip(names, EXPERIMENTS[args.experiment]):
        setattr(args, name, bool(value))
    if args.dataset_name != 'mimic_cxr':
        raise ValueError('This revision protocol supports MIMIC-CXR only.')
    if args.n_gpu != 1:
        raise ValueError('Use independent single-GPU experiments; multi-GPU training is not implemented.')
    if not args.use_dgl:
        args.cls_lambda = 0.0
    return args
