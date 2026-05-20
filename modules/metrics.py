import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'pycocoevalcap', 'pycocoevalcap'))

from pycocoevalcap.pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.pycocoevalcap.cider.cider import Cider


def compute_scores(gts, res):
    scorers = [
        (Bleu(4), ["BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]
    eval_res = {}
    
    for scorer, method in scorers:
        try:
            score, scores = scorer.compute_score(gts, res, verbose=0)
        except TypeError:
            score, scores = scorer.compute_score(gts, res)
            
        if isinstance(method, list):
            for sc, m in zip(score, method):
                eval_res[m] = sc
        else:
            eval_res[method] = score
            
    return eval_res