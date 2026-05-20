import os
import csv
import re
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import warnings
from transformers import BertTokenizer, BertModel, logging
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# ================= 配置区域 =================
CHEXBERT_PATH = '/home/fgy/dgl/Longitudinal/data/chexbert.pth'
BERT_LOCAL_PATH = '/home/fgy/dgl/Longitudinal/data/bert_local'
RESULT_DIR = '/home/fgy/dgl/Longitudinal/result'

# 输出目录
OUTPUT_DIR = '/home/fgy/dgl/Longitudinal/result/ce_category_results'

# 文件格式：
# auto: 自动识别
# dash: 10-gt_test.csv / 10-res_test.csv
# compact: 10gtstes.csv / 10restes.csv
FILE_STYLE = 'auto'

# 以哪个指标选最佳 epoch
# 可选：micro_f1 / macro_f1
BEST_METRIC = 'macro_f1'

warnings.filterwarnings("ignore")
logging.set_verbosity_error()
# ===========================================


# CheXbert 14 类标签顺序
# 注意：最后一类 No Finding 是二分类，其余 13 类是四分类转二值
CHEXBERT_LABELS = [
    'Enlarged Cardiomediastinum',
    'Cardiomegaly',
    'Lung Opacity',
    'Lung Lesion',
    'Edema',
    'Consolidation',
    'Pneumonia',
    'Atelectasis',
    'Pneumothorax',
    'Pleural Effusion',
    'Pleural Other',
    'Fracture',
    'Support Devices',
    'No Finding'
]


class CheXbert(nn.Module):
    def __init__(self):
        super(CheXbert, self).__init__()
        self.bert = BertModel.from_pretrained(BERT_LOCAL_PATH)
        self.linear_heads = nn.ModuleList()

        for _ in range(13):
            self.linear_heads.append(nn.Linear(self.bert.config.hidden_size, 4))

        self.linear_heads.append(nn.Linear(self.bert.config.hidden_size, 2))

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]

        logits_list = []
        for i in range(14):
            logits_list.append(self.linear_heads[i](cls_output))

        return logits_list


def read_single_column_csv(path):
    texts = []

    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                texts.append(row[0])

    return texts


def discover_epochs():
    """
    自动识别 epoch 文件。

    支持两种格式：
    1. dash:
       10-gt_test.csv / 10-res_test.csv

    2. compact:
       10gtstes.csv / 10restes.csv
    """
    if not os.path.exists(RESULT_DIR):
        print(f"❌ 错误: 目录不存在 {RESULT_DIR}")
        return []

    files = os.listdir(RESULT_DIR)
    found = {}

    for f in files:
        # 旧格式：10-gt_test.csv
        m_dash = re.match(r'^(\d+)-gt_test\.csv$', f)

        # 新格式：10gtstes.csv
        m_compact = re.match(r'^(\d+)gtstes\.csv$', f)

        if FILE_STYLE in ['auto', 'dash'] and m_dash:
            epoch = int(m_dash.group(1))
            gt_file = f"{epoch}-gt_test.csv"
            res_file = f"{epoch}-res_test.csv"

            if (
                os.path.exists(os.path.join(RESULT_DIR, gt_file))
                and os.path.exists(os.path.join(RESULT_DIR, res_file))
            ):
                found[epoch] = {
                    'style': 'dash',
                    'gt_file': gt_file,
                    'res_file': res_file
                }

        if FILE_STYLE in ['auto', 'compact'] and m_compact:
            epoch = int(m_compact.group(1))
            gt_file = f"{epoch}gtstes.csv"
            res_file = f"{epoch}restes.csv"

            if (
                os.path.exists(os.path.join(RESULT_DIR, gt_file))
                and os.path.exists(os.path.join(RESULT_DIR, res_file))
            ):
                found[epoch] = {
                    'style': 'compact',
                    'gt_file': gt_file,
                    'res_file': res_file
                }

    return [(epoch, found[epoch]) for epoch in sorted(found.keys())]


def load_data_from_csv(epoch, file_info):
    gt_path = os.path.join(RESULT_DIR, file_info['gt_file'])
    res_path = os.path.join(RESULT_DIR, file_info['res_file'])

    if not os.path.exists(gt_path) or not os.path.exists(res_path):
        print(f"⚠️ 警告: 找不到文件 {gt_path} 或 {res_path}")
        return None, None

    refs = read_single_column_csv(gt_path)
    hyps = read_single_column_csv(res_path)

    if len(refs) != len(hyps):
        print(
            f"⚠️ 警告: Epoch {epoch} 中 GT 与 RES 数量不一致: "
            f"GT={len(refs)}, RES={len(hyps)}，将截断到较短长度。"
        )
        min_len = min(len(refs), len(hyps))
        refs = refs[:min_len]
        hyps = hyps[:min_len]

    return refs, hyps


def get_binary_labels(texts, tokenizer, model, device, infer_batch_size=64):
    batch_preds = []

    for i in range(0, len(texts), infer_batch_size):
        batch = texts[i:i + infer_batch_size]

        inputs = tokenizer(
            batch,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits_list = model(inputs['input_ids'], inputs['attention_mask'])

            current_batch_preds = []
            for idx, logit in enumerate(logits_list):
                pred_class = logit.argmax(dim=1)

                # 前 13 类：
                # CheXbert 输出 4 类，通常将 positive / uncertain 映射为阳性
                if idx < 13:
                    binary = (pred_class == 1) | (pred_class == 3)

                # 最后一类 No Finding：
                # 二分类
                else:
                    binary = (pred_class == 1)

                current_batch_preds.append(binary.long().unsqueeze(1))

            batch_preds.append(torch.cat(current_batch_preds, dim=1))

    return torch.cat(batch_preds, dim=0).cpu().numpy()


def compute_ce_metrics(ref_labels, hyp_labels, epoch):
    # ========= 整体指标 =========
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
        ref_labels,
        hyp_labels,
        average='micro',
        zero_division=0
    )

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        ref_labels,
        hyp_labels,
        average='macro',
        zero_division=0
    )

    acc_element = np.mean(ref_labels == hyp_labels)
    acc_exact = accuracy_score(ref_labels, hyp_labels)

    summary = {
        'epoch': epoch,
        'acc_element': acc_element,
        'acc_exact': acc_exact,
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'micro_f1': micro_f1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1
    }

    # ========= 类别级指标 =========
    per_precision, per_recall, per_f1, per_support = precision_recall_fscore_support(
        ref_labels,
        hyp_labels,
        average=None,
        zero_division=0
    )

    per_class_rows = []

    for i, disease in enumerate(CHEXBERT_LABELS):
        gt_i = ref_labels[:, i]
        pred_i = hyp_labels[:, i]

        tp = int(((gt_i == 1) & (pred_i == 1)).sum())
        fp = int(((gt_i == 0) & (pred_i == 1)).sum())
        fn = int(((gt_i == 1) & (pred_i == 0)).sum())
        tn = int(((gt_i == 0) & (pred_i == 0)).sum())

        class_acc = float((gt_i == pred_i).mean())

        per_class_rows.append({
            'epoch': epoch,
            'class_index': i,
            'disease': disease,
            'support': int(per_support[i]),
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn,
            'accuracy': class_acc,
            'precision': float(per_precision[i]),
            'recall': float(per_recall[i]),
            'f1': float(per_f1[i])
        })

    per_class_df = pd.DataFrame(per_class_rows)

    return summary, per_class_df


def evaluate_all():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"🚀 开始批量评估 (Device: {device})")
    print(f"📂 结果目录: {RESULT_DIR}")
    print(f"📁 输出目录: {OUTPUT_DIR}")
    print(f"📄 文件格式模式: {FILE_STYLE}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 初始化 CheXbert
    print("\n⏳ 正在加载 CheXbert 模型...")
    tokenizer = BertTokenizer.from_pretrained(BERT_LOCAL_PATH)
    model = CheXbert()

    if not os.path.exists(CHEXBERT_PATH):
        print(f"❌ 错误: 找不到权重 {CHEXBERT_PATH}")
        return

    checkpoint = torch.load(CHEXBERT_PATH, map_location='cpu')
    state_dict = (
        checkpoint['model_state_dict']
        if 'model_state_dict' in checkpoint
        else checkpoint
    )
    new_state_dict = {
        k.replace('module.', ''): v
        for k, v in state_dict.items()
    }

    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)
    model.eval()

    # 2. 扫描 epoch 文件
    epoch_items = discover_epochs()

    if not epoch_items:
        print("❌ 未发现任何符合格式的结果文件。")
        print("   支持格式示例：")
        print("   旧格式: 10-gt_test.csv + 10-res_test.csv")
        print("   新格式: 10gtstes.csv + 10restes.csv")
        return

    print(f"\n📋 发现 {len(epoch_items)} 个完整 Epoch:")
    for epoch, info in epoch_items:
        print(f"   Epoch {epoch}: {info['gt_file']} + {info['res_file']} ({info['style']})")

    print("\n" + "=" * 125)
    print(
        f"{'Epoch':<6} | {'Acc(Elem)':<10} | {'Acc(Exact)':<10} | "
        f"{'Micro P':<9} | {'Micro R':<9} | {'Micro F1':<9} | "
        f"{'Macro P':<9} | {'Macro R':<9} | {'Macro F1':<9}"
    )
    print("-" * 125)

    all_summary = []

    # 3. 循环评估每个 epoch
    for epoch, file_info in epoch_items:
        refs, hyps = load_data_from_csv(epoch, file_info)

        if refs is None:
            continue

        ref_labels = get_binary_labels(refs, tokenizer, model, device)
        hyp_labels = get_binary_labels(hyps, tokenizer, model, device)
        
        # ======================================================
        # 🌟 截胡矩阵代码开始 (在这里把 0/1 矩阵存下来)
        # ======================================================
        matrix_dir = os.path.join(OUTPUT_DIR, "npy_matrices")
        os.makedirs(matrix_dir, exist_ok=True)
        
        # 保存真实标签矩阵 (GT) 和预测标签矩阵 (Pred)
        np.save(os.path.join(matrix_dir, f"gt_matrix_epoch_{epoch}.npy"), ref_labels)
        np.save(os.path.join(matrix_dir, f"pred_matrix_epoch_{epoch}.npy"), hyp_labels)
        # ======================================================
        # 🌟 截胡矩阵代码结束
        # ======================================================

        summary, per_class_df = compute_ce_metrics(
            ref_labels,
            hyp_labels,
            epoch
        )

        all_summary.append(summary)

        # 保存类别级结果
        per_class_path = os.path.join(
            OUTPUT_DIR,
            f"ce_per_class_epoch_{epoch}.csv"
        )
        per_class_df.to_csv(per_class_path, index=False, encoding='utf-8-sig')

        print(
            f"{epoch:<6} | "
            f"{summary['acc_element']:.5f}    | "
            f"{summary['acc_exact']:.5f}    | "
            f"{summary['micro_precision']:.5f}   | "
            f"{summary['micro_recall']:.5f}   | "
            f"{summary['micro_f1']:.5f}   | "
            f"{summary['macro_precision']:.5f}   | "
            f"{summary['macro_recall']:.5f}   | "
            f"{summary['macro_f1']:.5f}"
        )

    if not all_summary:
        print("❌ 没有成功评估任何 epoch。")
        return

    # 4. 保存整体 summary
    summary_df = pd.DataFrame(all_summary)
    summary_path = os.path.join(OUTPUT_DIR, "ce_summary.csv")
    summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')

    # 5. 选择最佳 epoch
    if BEST_METRIC not in summary_df.columns:
        print(f"⚠️ BEST_METRIC={BEST_METRIC} 不存在，默认使用 macro_f1。")
        best_metric = 'macro_f1'
    else:
        best_metric = BEST_METRIC

    best_row = summary_df.sort_values(best_metric, ascending=False).iloc[0]
    best_epoch = int(best_row['epoch'])

    print("=" * 125)
    print(f"\n🏆 【最佳轮次 Based on {best_metric}】: Epoch {best_epoch}")
    print(f"   Accuracy(Elem): {best_row['acc_element']:.5f}")
    print(f"   Accuracy(Exact): {best_row['acc_exact']:.5f}")
    print(f"   Micro Precision: {best_row['micro_precision']:.5f}")
    print(f"   Micro Recall:    {best_row['micro_recall']:.5f}")
    print(f"   Micro F1:        {best_row['micro_f1']:.5f}")
    print(f"   Macro Precision: {best_row['macro_precision']:.5f}")
    print(f"   Macro Recall:    {best_row['macro_recall']:.5f}")
    print(f"   Macro F1:        {best_row['macro_f1']:.5f}")

    # 6. 打印最佳 epoch 的类别级结果
    best_per_class_path = os.path.join(
        OUTPUT_DIR,
        f"ce_per_class_epoch_{best_epoch}.csv"
    )

    if os.path.exists(best_per_class_path):
        best_per_class_df = pd.read_csv(best_per_class_path)

        print("\n📌 最佳 epoch 的类别级 CE 指标：")
        print("-" * 100)
        print(
            f"{'Disease':<30} | {'Support':<8} | {'Precision':<10} | "
            f"{'Recall':<10} | {'F1':<10}"
        )
        print("-" * 100)

        for _, row in best_per_class_df.iterrows():
            print(
                f"{row['disease']:<30} | "
                f"{int(row['support']):<8} | "
                f"{row['precision']:.5f}    | "
                f"{row['recall']:.5f}    | "
                f"{row['f1']:.5f}"
            )

    print("\n📁 已保存：")
    print(f"   整体 CE 结果: {summary_path}")
    print(f"   类别级 CE 结果: {OUTPUT_DIR}/ce_per_class_epoch_*.csv")
    print("\n" + "=" * 125)


if __name__ == "__main__":
    evaluate_all()