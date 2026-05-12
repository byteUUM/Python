# ==========================================
# 基于贝叶斯方法的垃圾邮件分类与不确定性分析
# 完整实验代码 - 包含所有可视化图表与表格
# ==========================================

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score, log_loss
)
from sklearn.calibration import calibration_curve
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# ===== 全局设置 =====
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 11

SAVE_FIG = True  # 是否保存图片

# ==========================================
# 第2章 数据集构建与预处理
# ==========================================
np.random.seed(42)

# --- 共享词池（制造模糊性） ---
shared_words = [
    "free", "urgent", "new", "special", "update", "offer", "click",
    "account", "confirm", "verify", "access", "information", "important",
    "change", "required", "action", "response", "request", "submit",
    "download", "online", "service", "notification", "message", "send",
    "receive", "available", "today", "now", "limited", "exclusive"
]

spam_specific = [
    "win cash prize lottery million winner claim bonus",
    "viagra pills cheap pharmacy discount buy cheap order",
    "credit debt loan mortgage refinance interest rate",
    "click below unsubscribe trial deal guaranteed offer",
    "click here nigerian inheritance casino gambling poker",
    "prescription enlargement diet weight loss miracle cure",
    "congratulations million dollars work from home income",
    "profit investment returns bonus reward signup earn",
    "gift card voucher coupon redeem limited time only",
    "best price free money easy income no obligation risk",
]

ham_specific = [
    "meeting schedule project deadline team review discussion",
    "attached document report analysis quarterly results summary",
    "lunch dinner party birthday celebration invitation family",
    "conference video agenda minutes notes follow up action",
    "vacation holiday trip travel booking reservation hotel",
    "proposal feedback milestone completion timeline resource",
    "policy company department employee onboarding welcome",
    "workshop training session skill development learning",
    "client budget expense reimbursement approval request",
    "system maintenance deployment release production update",
]

noise_words = [
    "the", "is", "at", "which", "and", "on", "a", "to", "for", "of",
    "with", "in", "this", "that", "it", "are", "be", "was", "were", "been",
    "have", "has", "had", "will", "would", "could", "should", "may", "can", "do"
]


def generate_email(spam=True):
    """生成单封邮件"""
    parts = []
    # 共享词 1-3个
    parts.extend(np.random.choice(shared_words, size=np.random.randint(1, 4)).tolist())
    # 特定词 2-4个短语
    specific = spam_specific if spam else ham_specific
    chosen = np.random.choice(specific, size=np.random.randint(2, 5), replace=False)
    for phrase in chosen:
        parts.extend(phrase.split())
    # 噪声词
    parts.extend(np.random.choice(noise_words, size=np.random.randint(5, 20)).tolist())
    np.random.shuffle(parts)
    return " ".join(parts)


# 生成数据集（类别不均衡）
n_spam = 1500
n_ham = 2200
spam_emails = [generate_email(spam=True) for _ in range(n_spam)]
ham_emails = [generate_email(spam=False) for _ in range(n_ham)]

all_emails = spam_emails + ham_emails
all_labels = [1] * n_spam + [0] * n_ham  # 1=spam, 0=ham

print("=" * 65)
print("  基于贝叶斯方法的垃圾邮件分类与不确定性分析 - 实验报告")
print("=" * 65)

# ==========================================
# 表5-1 数据集基本信息
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    all_emails, all_labels, test_size=0.3, random_state=42, stratify=all_labels
)

tfidf_vectorizer = TfidfVectorizer(
    max_features=3000, stop_words='english', lowercase=True,
    ngram_range=(1, 2), min_df=3, max_df=0.90
)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

sparsity = (1 - X_train_tfidf.nnz / (X_train_tfidf.shape[0] * X_train_tfidf.shape[1])) * 100

data_info = pd.DataFrame({
    '指标': ['总样本数', '训练集样本数', '测试集样本数', '垃圾邮件数',
             '正常邮件数', '垃圾邮件比例(%)', 'TF-IDF特征维度', '特征矩阵稀疏度(%)'],
    '值': [len(all_emails), len(X_train), len(X_test), n_spam,
           n_ham, f"{n_spam / len(all_emails) * 100:.1f}",
           X_train_tfidf.shape[1], f"{sparsity:.2f}"]
})
print("\n" + "=" * 65)
print("表5-1 数据集基本信息")
print("=" * 65)
print(data_info.to_string(index=False))

# ==========================================
# 图2-1 数据集类别分布
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 柱状图
counts = [n_ham, n_spam]
labels = ['Ham (Normal)', 'Spam']
colors = ['#4CAF50', '#F44336']
bars = axes[0].bar(labels, counts, color=colors, edgecolor='black', width=0.5)
for bar, count in zip(bars, counts):
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
                 str(count), ha='center', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Sample Count', fontsize=12)
axes[0].set_title('Class Distribution (Bar Chart)', fontsize=13)
axes[0].set_ylim(0, max(counts) * 1.15)

# 饼图
axes[1].pie(counts, labels=labels, colors=colors, autopct='%1.1f%%',
            startangle=90, explode=[0, 0.05], shadow=True,
            textprops={'fontsize': 12})
axes[1].set_title('Class Distribution (Pie Chart)', fontsize=13)

plt.tight_layout()
if SAVE_FIG:
    plt.savefig('./fig2_1_class_distribution.png', bbox_inches='tight')
plt.show()
print("[图2-1] 数据集类别分布图已生成")

# ==========================================
# 第5章 模型训练与对比实验
# ==========================================
print("\n" + "=" * 65)
print("第5章 实验设计与实现")
print("=" * 65)

models = {
    'Multinomial NB': MultinomialNB(alpha=1.0),
    'Complement NB': ComplementNB(alpha=1.0),
    'Bernoulli NB': BernoulliNB(alpha=1.0),
    'Logistic Regression': LogisticRegression(max_iter=1000, C=1.0, random_state=42),
    'SVM (RBF)': SVC(kernel='rbf', probability=True, C=1.0, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
}

results = {}
for name, model in models.items():
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    y_prob = model.predict_proba(X_test_tfidf)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    ll = log_loss(y_test, y_prob)
    auc_val = auc(*roc_curve(y_test, y_prob)[:2])

    results[name] = {
        'model': model, 'y_pred': y_pred, 'y_prob': y_prob,
        'accuracy': acc, 'precision': prec, 'recall': rec,
        'f1_score': f1, 'log_loss': ll, 'auc': auc_val
    }
    print(f"  {name:25s}  Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}  AUC={auc_val:.4f}")

# ==========================================
# 表6-1 分类性能评价指标对比
# ==========================================
metrics_df = pd.DataFrame({
    name: {
        'Accuracy': r['accuracy'],
        'Precision': r['precision'],
        'Recall': r['recall'],
        'F1-Score': r['f1_score'],
        'AUC': r['auc'],
        'Log Loss': r['log_loss']
    } for name, r in results.items()
}).T
metrics_df = metrics_df.round(4)

print("\n" + "=" * 65)
print("表6-1 各模型分类性能评价指标对比")
print("=" * 65)
print(metrics_df.to_string())

# ==========================================
# 图6-1 性能指标对比柱状图
# ==========================================
fig, ax = plt.subplots(figsize=(14, 6))
metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
x = np.arange(len(models))
width = 0.18
colors_bar = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']

for i, metric in enumerate(metrics_to_plot):
    vals = [results[name][metric.lower().replace('-', '_').replace('score', 'score')]
            for name in models.keys()]
    # 修正key
    if metric == 'F1-Score':
        vals = [results[name]['f1_score'] for name in models.keys()]
    elif metric == 'Accuracy':
        vals = [results[name]['accuracy'] for name in models.keys()]
    elif metric == 'Precision':
        vals = [results[name]['precision'] for name in models.keys()]
    elif metric == 'Recall':
        vals = [results[name]['recall'] for name in models.keys()]

    bars = ax.bar(x + i * width, vals, width, label=metric, color=colors_bar[i], edgecolor='black', linewidth=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{v:.3f}', ha='center', va='bottom', fontsize=8, rotation=45)

ax.set_xlabel('Model', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Table 6-1: Classification Performance Comparison', fontsize=14)
ax.set_xticks(x + 1.5 * width)
ax.set_xticklabels(list(models.keys()), rotation=15, ha='right')
ax.set_ylim(0.7, 1.05)
ax.legend(loc='lower right', fontsize=10)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
if SAVE_FIG:
    plt.savefig('./fig6_1_performance_comparison.png', bbox_inches='tight')
plt.show()
print("[图6-1] 模型性能对比柱状图已生成")

# ==========================================
# 图6-2 混淆矩阵 (2x3子图)
# ==========================================
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for idx, (name, r) in enumerate(results.items()):
    cm = confusion_matrix(y_test, r['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'],
                annot_kws={'size': 14})
    axes[idx].set_title(name, fontsize=12, fontweight='bold')
    axes[idx].set_ylabel('True Label', fontsize=10)
    axes[idx].set_xlabel('Predicted Label', fontsize=10)

plt.suptitle('Figure 6-2: Confusion Matrices for All Models', fontsize=14, y=1.02)
plt.tight_layout()
if SAVE_FIG:
    plt.savefig('./fig6_2_confusion_matrices.png', bbox_inches='tight')
plt.show()
print("[图6-2] 混淆矩阵图已生成")

# ==========================================
# 表6-2 Multinomial NB 混淆矩阵详细数据
# ==========================================
nb_cm = confusion_matrix(y_test, results['Multinomial NB']['y_pred'])
cm_detail = pd.DataFrame(
    nb_cm,
    index=['Actual Ham', 'Actual Spam'],
    columns=['Predicted Ham', 'Predicted Spam']
)
print("\n" + "=" * 65)
print("表6-2 Multinomial NB 混淆矩阵")
print("=" * 65)
print(cm_detail)

tn, fp, fn, tp = nb_cm.ravel()
print(f"\n  TN={tn}, FP={fp}, FN={fn}, TP={tp}")
print(f"  False Positive Rate (FPR): {fp / (fp + tn):.4f}")
print(f"  False Negative Rate (FNR): {fn / (fn + tp):.4f}")

# ==========================================
# 图6-3 ROC曲线对比
# ==========================================
fig, ax = plt.subplots(figsize=(10, 8))

colors_roc = ['#E53935', '#8E24AA', '#1E88E5', '#43A047', '#FB8C00', '#00ACC1']
for idx, (name, r) in enumerate(results.items()):
    fpr, tpr, _ = roc_curve(y_test, r['y_prob'])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=colors_roc[idx], lw=2.5,
            label=f'{name} (AUC = {roc_auc:.4f})')

ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Classifier')
ax.fill_between([0, 1], [0, 1], alpha=0.1, color='gray')
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])
ax.set_xlabel('False Positive Rate (FPR)', fontsize=13)
ax.set_ylabel('True Positive Rate (TPR)', fontsize=13)
ax.set_title('Figure 6-3: ROC Curve Comparison', fontsize=14)
ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
if SAVE_FIG:
    plt.savefig('./fig6_3_roc_curves.png', bbox_inches='tight')
plt.show()
print("[图6-3] ROC曲线对比图已生成")

# ==========================================
# 图6-4 Precision-Recall曲线
# ==========================================
fig, ax = plt.subplots(figsize=(10, 8))

for idx, (name, r) in enumerate(results.items()):
    prec_curve, rec_curve, _ = precision_recall_curve(y_test, r['y_prob'])
    ap = average_precision_score(y_test, r['y_prob'])
    ax.plot(rec_curve, prec_curve, color=colors_roc[idx], lw=2.5,
            label=f'{name} (AP = {ap:.4f})')

ax.set_xlim([-0.02, 1.02])
ax.set_ylim([0.5, 1.02])
ax.set_xlabel('Recall', fontsize=13)
ax.set_ylabel('Precision', fontsize=13)
ax.set_title('Figure 6-4: Precision-Recall Curve Comparison', fontsize=14)
ax.legend(loc='lower left', fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
if SAVE_FIG:
    plt.savefig('./fig6_4_pr_curves.png', bbox_inches='tight')
plt.show()
print("[图6-4] Precision-Recall曲线已生成")

# ==========================================
# 第7章 不确定性分析（重点章节）
# ==========================================
print("\n" + "=" * 65)
print("第7章 不确定性分析")
print("=" * 65)

# 7.1 概率输出的解释意义
nb_model = results['Multinomial NB']['model']
nb_prob = results['Multinomial NB']['y_prob']
nb_pred = results['Multinomial NB']['y_pred']

# ==========================================
# 图7-1 预测概率分布直方图
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 所有样本的P(spam)分布
axes[0].hist(nb_prob, bins=50, color='#42A5F5', edgecolor='black', alpha=0.7)
axes[0].axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Decision Boundary (0.5)')
axes[0].axvspan(0.3, 0.7, alpha=0.15, color='orange', label='Uncertain Zone [0.3, 0.7]')
axes[0].set_xlabel('P(Spam)', fontsize=12)
axes[0].set_ylabel('Count', fontsize=12)
axes[0].set_title('7-1a: P(Spam) Distribution (All Samples)', fontsize=13)
axes[0].legend(fontsize=9)

# 按真实标签分组
spam_probs = nb_prob[np.array(y_test) == 1]
ham_probs = nb_prob[np.array(y_test) == 0]
axes[1].hist(ham_probs, bins=50, color='#4CAF50', edgecolor='black', alpha=0.6, label='True Ham')
axes[1].hist(spam_probs, bins=50, color='#F44336', edgecolor='black', alpha=0.6, label='True Spam')
axes[1].axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Threshold = 0.5')
axes[1].axvspan(0.3, 0.7, alpha=0.15, color='orange', label='Uncertain Zone')
axes[1].set_xlabel('P(Spam)', fontsize=12)
axes[1].set_ylabel('Count', fontsize=12)
axes[1].set_title('7-1b: P(Spam) Distribution by True Label', fontsize=13)
axes[1].legend(fontsize=9)

plt.tight_layout()
if SAVE_FIG:
    plt.savefig('./fig7_1_probability_distribution.png', bbox_inches='tight')
plt.show()
print("[图7-1] 预测概率分布直方图已生成")

# ==========================================
# 表7-1 概率区间统计
# ==========================================
prob_ranges = [
    ('[0.0, 0.1)', (0.0, 0.1)),
    ('[0.1, 0.2)', (0.1, 0.2)),
    ('[0.2, 0.3)', (0.2, 0.3)),
    ('[0.3, 0.4)', (0.3, 0.4)),
    ('[0.4, 0.5)', (0.4, 0.5)),
    ('[0.5, 0.6)', (0.5, 0.6)),
    ('[0.6, 0.7)', (0.6, 0.7)),
    ('[0.7, 0.8)', (0.7, 0.8)),
    ('[0.8, 0.9)', (0.8, 0.9)),
    ('[0.9, 1.0]', (0.9, 1.01)),
]

y_test_arr = np.array(y_test)
prob_stats = []
for label, (lo, hi) in prob_ranges:
    mask = (nb_prob >= lo) & (nb_prob < hi)
    n_total = mask.sum()
    n_spam_in = ((y_test_arr == 1) & mask).sum()
    n_ham_in = ((y_test_arr == 0) & mask).sum()
    prob_stats.append({
        'P(Spam) Range': label,
        'Total Samples': n_total,
        'True Spam': n_spam_in,
        'True Ham': n_ham_in,
        'Spam Ratio (%)': f"{n_spam_in / n_total * 100:.1f}" if n_total > 0 else '-',
    })

prob_stats_df = pd.DataFrame(prob_stats)
print("\n" + "=" * 65)
print("表7-1 预测概率区间样本分布统计")
print("=" * 65)
print(prob_stats_df.to_string(index=False))

# ==========================================
# 7.2 分类置信度分析
# ==========================================
confidence = np.abs(nb_prob - 0.5) * 2  # 映射到[0,1], 1=最高置信度

# ==========================================
# 图7-2 置信度分布与正确率关系
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 置信度分布
axes[0].hist(confidence, bins=50, color='#7E57C2', edgecolor='black', alpha=0.7)
axes[0].axvline(x=0.4, color='red', linestyle='--', linewidth=2, label='Low Confidence Threshold (0.4)')
axes[0].set_xlabel('Confidence Score', fontsize=12)
axes[0].set_ylabel('Count', fontsize=12)
axes[0].set_title('7-2a: Confidence Score Distribution', fontsize=13)
axes[0].legend(fontsize=9)

# 不同置信度区间的准确率
correct = (nb_pred == y_test_arr)
bins_edges = np.linspace(0, 1, 11)
bin_accs = []
bin_labels = []
for i in range(len(bins_edges) - 1):
    mask = (confidence >= bins_edges[i]) & (confidence < bins_edges[i + 1])
    if mask.sum() > 0:
        acc_bin = correct[mask].mean()
        bin_accs.append(acc_bin)
        bin_labels.append(f'{bins_edges[i]:.1f}-{bins_edges[i + 1]:.1f}')

bars = axes[1].bar(range(len(bin_accs)), bin_accs, color='#26A69A', edgecolor='black')
axes[1].set_xticks(range(len(bin_labels)))
axes[1].set_xticklabels(bin_labels, rotation=45, ha='right')
axes[1].set_ylabel('Accuracy', fontsize=12)
axes[1].set_xlabel('Confidence Range', fontsize=12)
axes[1].set_title('7-2b: Accuracy by Confidence Level', fontsize=13)
axes[1].set_ylim(0.5, 1.05)
for bar, v in zip(bars, bin_accs):
    axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{v:.3f}', ha='center', fontsize=9)

plt.tight_layout()
if SAVE_FIG:
    plt.savefig('./fig7_2_confidence_analysis.png', bbox_inches='tight')
plt.show()
print("[图7-2] 置信度分析图已生成")

# ==========================================
# 表7-2 不同置信度水平的分类统计
# ==========================================
confidence_levels = [
    ('Very Low [0.0-0.2)', 0.0, 0.2),
    ('Low [0.2-0.4)', 0.2, 0.4),
    ('Medium [0.4-0.6)', 0.4, 0.6),
    ('High [0.6-0.8)', 0.6, 0.8),
    ('Very High [0.8-1.0]', 0.8, 1.01),
]

conf_stats = []
for label, lo, hi in confidence_levels:
    mask = (confidence >= lo) & (confidence < hi)
    n = mask.sum()
    if n > 0:
        acc = correct[mask].mean()
        n_fp = ((nb_pred != y_test_arr) & (y_test_arr == 0) & mask).sum()
        n_fn = ((nb_pred != y_test_arr) & (y_test_arr == 1) & mask).sum()
        conf_stats.append({
            'Confidence Level': label,
            'Sample Count': n,
            'Accuracy': f'{acc:.4f}',
            'False Positive': n_fp,
            'False Negative': n_fn,
            'Ratio (%)': f'{n / len(confidence) * 100:.1f}'
        })

conf_df = pd.DataFrame(conf_stats)
print("\n" + "=" * 65)
print("表7-2 不同置信度水平分类统计")
print("=" * 65)
print(conf_df.to_string(index=False))

# ==========================================
# 7.3 不确定样本案例分析
# ==========================================
uncertain_threshold = 0.3
uncertain_mask = np.abs(nb_prob - 0.5) < uncertain_threshold
uncertain_indices = np.where(uncertain_mask)[0]

print(f"\n不确定样本数量 (|P(spam)-0.5| < {uncertain_threshold}): {len(uncertain_indices)}")
print(f"占总测试样本比例: {len(uncertain_indices) / len(y_test) * 100:.2f}%")

# 不确定样本的误判率
if len(uncertain_indices) > 0:
    uncertain_correct = correct[uncertain_indices].mean()
    certain_mask = ~uncertain_mask
    certain_correct = correct[certain_mask].mean()
    print(f"不确定样本准确率: {uncertain_correct:.4f}")
    print(f"确定样本准确率:   {certain_correct:.4f}")

# ==========================================
# 图7-3 不确定样本vs确定样本分析
# ==========================================
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# (a) 不确定样本的P(spam)散点图
sample_idx = np.arange(len(y_test))
scatter_colors = ['#4CAF50' if y == 0 else '#F44336' for y in y_test_arr]

axes[0].scatter(sample_idx[uncertain_mask], nb_prob[uncertain_mask],
                c=[scatter_colors[i] for i in range(len(scatter_colors)) if uncertain_mask[i]],
                alpha=0.6, s=20, edgecolors='black', linewidth=0.3)
axes[0].axhline(y=0.5, color='blue', linestyle='--', linewidth=1.5)
axes[0].axhspan(0.5 - uncertain_threshold, 0.5 + uncertain_threshold,
                alpha=0.1, color='orange', label='Uncertain Zone')
axes[0].set_xlabel('Sample Index', fontsize=12)
axes[0].set_ylabel('P(Spam)', fontsize=12)
axes[0].set_title('7-3a: Uncertain Samples Distribution', fontsize=13)
axes[0].legend(fontsize=9)

# (b) 箱线图比较
box_data = pd.DataFrame({
    'Confidence': ['Uncertain' if uncertain_mask[i] else 'Certain' for i in range(len(y_test))],
    'P(Spam)': nb_prob,
    'Correct': correct
})
sns.boxplot(x='Confidence', y='P(Spam)', data=box_data, ax=axes[1],
            palette=['#FF7043', '#42A5F5'])
axes[1].set_title('7-3b: P(Spam) by Certainty', fontsize=13)

# (c) 不确定/确定样本准确率对比
categories = ['Uncertain\nSamples', 'Certain\nSamples']
accs_compare = [uncertain_correct if len(uncertain_indices) > 0 else 0, certain_correct]
bar_colors = ['#FF7043', '#42A5F5']
bars = axes[2].bar(categories, accs_compare, color=bar_colors, edgecolor='black', width=0.5)
for bar, v in zip(bars, accs_compare):
    axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{v:.4f}', ha='center', fontsize=12, fontweight='bold')
axes[2].set_ylabel('Accuracy', fontsize=12)
axes[2].set_title('7-3c: Accuracy Comparison', fontsize=13)
axes[2].set_ylim(0, 1.1)

plt.tight_layout()
if SAVE_FIG:
    plt.savefig('./fig7_3_uncertain_analysis.png', bbox_inches='tight')
plt.show()
print("[图7-3] 不确定样本分析图已生成")

# ==========================================
# 表7-3 不确定样本详细案例
# ==========================================
if len(uncertain_indices) > 0:
    uncertain_cases = []
    sorted_uncertain = sorted(uncertain_indices, key=lambda i: abs(nb_prob[i] - 0.5))
    for i in sorted_uncertain[:15]:
        uncertain_cases.append({
            'Sample Index': i,
            'P(Spam)': f'{nb_prob[i]:.4f}',
            'Predicted': 'Spam' if nb_pred[i] == 1 else 'Ham',
            'Actual': 'Spam' if y_test_arr[i] == 1 else 'Ham',
            'Correct': '✓' if nb_pred[i] == y_test_arr[i] else '✗',
            'Uncertainty': f'{abs(nb_prob[i] - 0.5):.4f}',
            'Email (truncated)': X_test[i][:80] + '...'
        })

    cases_df = pd.DataFrame(uncertain_cases)
    print("\n" + "=" * 65)
    print("表7-3 不确定样本详细案例（Top 15 most uncertain）")
    print("=" * 65)
    for _, row in cases_df.iterrows():
        print(f"  [{row['Sample Index']:3d}] P(spam)={row['P(Spam)']}  "
              f"Pred={row['Predicted']:4s}  Actual={row['Actual']:4s}  "
              f"{'✓' if row['Correct'] == '✓' else '✗ Miss!'}  "
              f"Uncertainty={row['Uncertainty']}")
        print(f"        Text: {row['Email (truncated)']}")

# ==========================================
# 图7-4 概率校准曲线
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# (a) 校准曲线
for idx, (name, r) in enumerate(results.items()):
    fraction_pos, mean_predicted = calibration_curve(
        y_test, r['y_prob'], n_bins=10, strategy='uniform'
    )
    axes[0].plot(mean_predicted, fraction_pos, 's-', color=colors_roc[idx],
                 lw=2, markersize=5, label=name)

axes[0].plot([0, 1], [0, 1], 'k--', lw=1.5, label='Perfectly Calibrated')
axes[0].set_xlabel('Mean Predicted Probability', fontsize=12)
axes[0].set_ylabel('Fraction of Positives', fontsize=12)
axes[0].set_title('7-4a: Calibration Curve (Reliability Diagram)', fontsize=13)
axes[0].legend(loc='lower right', fontsize=8)
axes[0].set_xlim([-0.02, 1.02])
axes[0].set_ylim([-0.02, 1.02])

# (b) 各模型概率分布对比
for idx, (name, r) in enumerate(results.items()):
    axes[1].hist(r['y_prob'], bins=50, alpha=0.4, color=colors_roc[idx],
                 label=name, density=True)
axes[1].axvline(x=0.5, color='black', linestyle='--', linewidth=2)
axes[1].set_xlabel('P(Spam)', fontsize=12)
axes[1].set_ylabel('Density', fontsize=12)
axes[1].set_title('7-4b: Probability Distribution Comparison', fontsize=13)
axes[1].legend(fontsize=8)

plt.tight_layout()
if SAVE_FIG:
    plt.savefig('./fig7_4_calibration.png', bbox_inches='tight')
plt.show()
print("[图7-4] 概率校准曲线已生成")

# ==========================================
# 图7-5 贝叶斯模型不确定性 vs 其他模型
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# (a) 熵分析
nb_entropy = -nb_prob * np.log2(nb_prob + 1e-10) - (1 - nb_prob) * np.log2(1 - nb_prob + 1e-10)
lr_prob = results['Logistic Regression']['y_prob']
lr_entropy = -lr_prob * np.log2(lr_prob + 1e-10) - (1 - lr_prob) * np.log2(1 - lr_prob + 1e-10)

axes[0].hist(nb_entropy, bins=50, alpha=0.6, color='#E53935', label='Multinomial NB', density=True)
axes[0].hist(lr_entropy, bins=50, alpha=0.6, color='#1E88E5', label='Logistic Regression', density=True)
axes[0].set_xlabel('Prediction Entropy (bits)', fontsize=12)
axes[0].set_ylabel('Density', fontsize=12)
axes[0].set_title('7-5a: Prediction Entropy Distribution', fontsize=13)
axes[0].legend(fontsize=10)

# (b) 不同阈值下的决策风险
thresholds = np.arange(0.1, 0.9, 0.05)
nb_risks = []
lr_risks = []
for t in thresholds:
    nb_custom_pred = (nb_prob >= t).astype(int)
    lr_custom_pred = (lr_prob >= t).astype(int)
    # 风险 = 加权误判率 (FP权重更高)
    nb_fp = ((nb_custom_pred == 1) & (y_test_arr == 0)).sum()
    nb_fn = ((nb_custom_pred == 0) & (y_test_arr == 1)).sum()
    lr_fp = ((lr_custom_pred == 1) & (y_test_arr == 0)).sum()
    lr_fn = ((lr_custom_pred == 0) & (y_test_arr == 1)).sum()
    # 假设FP代价=5, FN代价=1
    nb_risk = (5 * nb_fp + 1 * nb_fn) / len(y_test)
    lr_risk = (5 * lr_fp + 1 * lr_fn) / len(y_test)
    nb_risks.append(nb_risk)
    lr_risks.append(lr_risk)

axes[1].plot(thresholds, nb_risks, 'o-', color='#E53935', lw=2, markersize=4, label='Multinomial NB')
axes[1].plot(thresholds, lr_risks, 's-', color='#1E88E5', lw=2, markersize=4, label='Logistic Regression')
axes[1].axvline(x=0.5, color='green', linestyle='--', lw=1.5, label='Default Threshold=0.5')

# 找最优阈值
opt_idx_nb = np.argmin(nb_risks)
opt_idx_lr = np.argmin(lr_risks)
axes[1].axvline(x=thresholds[opt_idx_nb], color='#E53935', linestyle=':', lw=1.5,
                label=f'NB Optimal t={thresholds[opt_idx_nb]:.2f}')
axes[1].axvline(x=thresholds[opt_idx_lr], color='#1E88E5', linestyle=':', lw=1.5,
                label=f'LR Optimal t={thresholds[opt_idx_lr]:.2f}')

axes[1].set_xlabel('Decision Threshold', fontsize=12)
axes[1].set_ylabel('Weighted Risk (FP cost=5, FN cost=1)', fontsize=11)
axes[1].set_title('7-5b: Decision Risk vs Threshold', fontsize=13)
axes[1].legend(fontsize=9)

plt.tight_layout()
if SAVE_FIG:
    plt.savefig('./fig7_5_uncertainty_entropy_risk.png', bbox_inches='tight')
plt.show()
print("[图7-5] 熵与决策风险分析图已生成")

# ==========================================
# 图7-6 交叉验证稳定性分析
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# (a) 交叉验证分数箱线图
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cv_results = {}
for name in ['Multinomial NB', 'Logistic Regression', 'SVM (RBF)', 'Random Forest']:
    model = models[name]
    scores = cross_val_score(model, X_train_tfidf, y_train, cv=cv, scoring='f1')
    cv_results[name] = scores

cv_df = pd.DataFrame(cv_results)
sns.boxplot(data=cv_df, ax=axes[0], palette=['#E53935', '#1E88E5', '#43A047', '#FB8C00'])
axes[0].set_ylabel('F1-Score', fontsize=12)
axes[0].set_title('7-6a: 10-Fold CV F1-Score Stability', fontsize=13)
axes[0].tick_params(axis='x', rotation=15)

# (b) 不同训练集大小下的学习曲线
train_sizes = np.arange(0.1, 1.01, 0.1)
nb_train_scores = []
nb_test_scores = []
lr_train_scores = []
lr_test_scores = []

for frac in train_sizes:
    n = int(len(X_train_tfidf.toarray()) * frac)
    X_sub = X_train_tfidf[:n]
    y_sub = y_train[:n]

    nb_temp = MultinomialNB(alpha=1.0).fit(X_sub, y_sub)
    nb_train_scores.append(f1_score(y_sub, nb_temp.predict(X_sub)))
    nb_test_scores.append(f1_score(y_test, nb_temp.predict(X_test_tfidf)))

    lr_temp = LogisticRegression(max_iter=1000, random_state=42).fit(X_sub, y_sub)
    lr_train_scores.append(f1_score(y_sub, lr_temp.predict(X_sub)))
    lr_test_scores.append(f1_score(y_test, lr_temp.predict(X_test_tfidf)))

axes[1].plot(train_sizes * 100, nb_train_scores, 'o--', color='#E53935', alpha=0.7, label='NB Train')
axes[1].plot(train_sizes * 100, nb_test_scores, 'o-', color='#E53935', lw=2, label='NB Test')
axes[1].plot(train_sizes * 100, lr_train_scores, 's--', color='#1E88E5', alpha=0.7, label='LR Train')
axes[1].plot(train_sizes * 100, lr_test_scores, 's-', color='#1E88E5', lw=2, label='LR Test')
axes[1].set_xlabel('Training Set Size (%)', fontsize=12)
axes[1].set_ylabel('F1-Score', fontsize=12)
axes[1].set_title('7-6b: Learning Curve', fontsize=13)
axes[1].legend(fontsize=9)
axes[1].set_ylim(0.5, 1.05)

plt.tight_layout()
if SAVE_FIG:
    plt.savefig('./fig7_6_cv_stability.png', bbox_inches='tight')
plt.show()
print("[图7-6] 交叉验证与学习曲线已生成")

# ==========================================
# 图7-7 拉普拉斯平滑参数影响分析
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

alphas = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
alpha_accs = []
alpha_f1s = []
alpha_lls = []
alpha_uncertain_ratios = []

for alpha in alphas:
    temp_nb = MultinomialNB(alpha=alpha).fit(X_train_tfidf, y_train)
    temp_pred = temp_nb.predict(X_test_tfidf)
    temp_prob = temp_nb.predict_proba(X_test_tfidf)[:, 1]

    alpha_accs.append(accuracy_score(y_test, temp_pred))
    alpha_f1s.append(f1_score(y_test, temp_pred))
    alpha_lls.append(log_loss(y_test, temp_prob))
    # 不确定样本比例
    uncertain_ratio = (np.abs(temp_prob - 0.5) < 0.2).mean()
    alpha_uncertain_ratios.append(uncertain_ratio)

axes[0].plot(alphas, alpha_accs, 'o-', color='#2196F3', lw=2, label='Accuracy')
axes[0].plot(alphas, alpha_f1s, 's-', color='#4CAF50', lw=2, label='F1-Score')
axes[0].set_xscale('log')
axes[0].axvline(x=1.0, color='red', linestyle='--', lw=1.5, label='Default alpha=1.0')
axes[0].set_xlabel('Laplace Smoothing Parameter (alpha)', fontsize=12)
axes[0].set_ylabel('Score', fontsize=12)
axes[0].set_title('7-7a: Performance vs Smoothing Parameter', fontsize=13)
axes[0].legend(fontsize=10)

ax2 = axes[1]
ax2.plot(alphas, alpha_lls, 'D-', color='#FF9800', lw=2, label='Log Loss')
ax2.set_xscale('log')
ax2.set_xlabel('Laplace Smoothing Parameter (alpha)', fontsize=12)
ax2.set_ylabel('Log Loss', fontsize=12, color='#FF9800')
ax2.tick_params(axis='y', labelcolor='#FF9800')

ax3 = ax2.twinx()
ax3.plot(alphas, alpha_uncertain_ratios, '^--', color='#9C27B0', lw=2, label='Uncertain Ratio')
ax3.set_ylabel('Uncertain Sample Ratio', fontsize=12, color='#9C27B0')
ax3.tick_params(axis='y', labelcolor='#9C27B0')
ax2.axvline(x=1.0, color='red', linestyle='--', lw=1.5)
ax2.set_title('7-7b: Log Loss & Uncertainty vs Smoothing', fontsize=13)

lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax3.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=10)

plt.tight_layout()
if SAVE_FIG:
    plt.savefig('./fig7_7_smoothing_analysis.png', bbox_inches='tight')
plt.show()
print("[图7-7] 拉普拉斯平滑参数分析图已生成")

# ==========================================
# 表7-4 拉普拉斯平滑参数影响
# ==========================================
smooth_df = pd.DataFrame({
    'alpha': alphas,
    'Accuracy': [f'{v:.4f}' for v in alpha_accs],
    'F1-Score': [f'{v:.4f}' for v in alpha_f1s],
    'Log Loss': [f'{v:.4f}' for v in alpha_lls],
    'Uncertain Ratio': [f'{v:.4f}' for v in alpha_uncertain_ratios]
})
print("\n" + "=" * 65)
print("表7-4 拉普拉斯平滑参数对模型性能与不确定性的影响")
print("=" * 65)
print(smooth_df.to_string(index=False))

# ==========================================
# 图7-8 特征重要性 - 朴素贝叶斯关键词分析
# ==========================================
feature_names = tfidf_vectorizer.get_feature_names_out()
log_prob_diff = nb_model.feature_log_prob_[1] - nb_model.feature_log_prob_[0]  # spam - ham

# 最具区分度的词
top_spam_idx = np.argsort(log_prob_diff)[-20:]
top_ham_idx = np.argsort(log_prob_diff)[:20]

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Spam指示词
spam_words = [feature_names[i] for i in top_spam_idx]
spam_scores = [log_prob_diff[i] for i in top_spam_idx]
axes[0].barh(range(len(spam_words)), spam_scores, color='#F44336', edgecolor='black', alpha=0.8)
axes[0].set_yticks(range(len(spam_words)))
axes[0].set_yticklabels(spam_words, fontsize=9)
axes[0].set_xlabel('Log Probability Difference (Spam - Ham)', fontsize=11)
axes[0].set_title('Top 20 Spam-Indicative Words', fontsize=13, fontweight='bold')

# Ham指示词
ham_words = [feature_names[i] for i in top_ham_idx]
ham_scores = [log_prob_diff[i] for i in top_ham_idx]
axes[1].barh(range(len(ham_words)), ham_scores, color='#4CAF50', edgecolor='black', alpha=0.8)
axes[1].set_yticks(range(len(ham_words)))
axes[1].set_yticklabels(ham_words, fontsize=9)
axes[1].set_xlabel('Log Probability Difference (Spam - Ham)', fontsize=11)
axes[1].set_title('Top 20 Ham-Indicative Words', fontsize=13, fontweight='bold')

plt.suptitle('Figure 7-8: Feature Importance Analysis (Naive Bayes)', fontsize=14, y=1.02)
plt.tight_layout()
if SAVE_FIG:
    plt.savefig('./fig7_8_feature_importance.png', bbox_inches='tight')
plt.show()
print("[图7-8] 特征重要性（关键词分析）图已生成")

# ==========================================
# 图7-9 分层决策机制示意图
# ==========================================
fig, ax = plt.subplots(figsize=(12, 7))

# 按置信度分层的决策流程
decision_categories = []
nb_confidence_vals = np.abs(nb_prob - 0.5) * 2

high_conf_mask = nb_confidence_vals >= 0.8
medium_conf_mask = (nb_confidence_vals >= 0.4) & (nb_confidence_vals < 0.8)
low_conf_mask = nb_confidence_vals < 0.4

n_high = high_conf_mask.sum()
n_med = medium_conf_mask.sum()
n_low = low_conf_mask.sum()

acc_high = correct[high_conf_mask].mean() if n_high > 0 else 0
acc_med = correct[medium_conf_mask].mean() if n_med > 0 else 0
acc_low = correct[low_conf_mask].mean() if n_low > 0 else 0

# 可视化分层决策
categories = ['High Confidence\n(Auto-Classify)', 'Medium Confidence\n(Flag Review)', 'Low Confidence\n(Human Review)']
counts = [n_high, n_med, n_low]
accs = [acc_high, acc_med, acc_low]
colors = ['#4CAF50', '#FF9800', '#F44336']

bars1 = ax.bar(np.arange(3) - 0.15, counts, 0.3, color=colors, edgecolor='black', alpha=0.7, label='Sample Count')
ax2 = ax.twinx()
bars2 = ax2.bar(np.arange(3) + 0.15, accs, 0.3, color=colors, edgecolor='black', alpha=0.4,
                hatch='//', label='Accuracy')

for i, (c, a) in enumerate(zip(counts, accs)):
    ax.text(i - 0.15, c + 10, str(c), ha='center', fontsize=11, fontweight='bold')
    ax2.text(i + 0.15, a + 0.01, f'{a:.3f}', ha='center', fontsize=11, fontweight='bold')

ax.set_xticks(range(3))
ax.set_xticklabels(categories, fontsize=12)
ax.set_ylabel('Sample Count', fontsize=12)
ax2.set_ylabel('Accuracy', fontsize=12, color='gray')
ax2.set_ylim(0, 1.1)
ax.set_title('Figure 7-9: Tiered Decision Mechanism Based on Confidence', fontsize=14)

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc='upper right')

plt.tight_layout()
if SAVE_FIG:
    plt.savefig('./fig7_9_tiered_decision.png', bbox_inches='tight')
plt.show()
print("[图7-9] 分层决策机制图已生成")

# ==========================================
# 表7-5 分层决策机制统计
# ==========================================
tiered_df = pd.DataFrame({
    'Decision Tier': ['High Confidence (Auto)', 'Medium Confidence (Flag)', 'Low Confidence (Human)'],
    'Confidence Range': ['[0.8, 1.0]', '[0.4, 0.8)', '[0.0, 0.4)'],
    'Sample Count': [n_high, n_med, n_low],
    'Ratio (%)': [f'{n_high / len(y_test) * 100:.1f}', f'{n_med / len(y_test) * 100:.1f}',
                  f'{n_low / len(y_test) * 100:.1f}'],
    'Accuracy': [f'{acc_high:.4f}', f'{acc_med:.4f}', f'{acc_low:.4f}'],
    'FP Count': [((nb_pred != y_test_arr) & (y_test_arr == 0) & high_conf_mask).sum(),
                 ((nb_pred != y_test_arr) & (y_test_arr == 0) & medium_conf_mask).sum(),
                 ((nb_pred != y_test_arr) & (y_test_arr == 0) & low_conf_mask).sum()],
    'FN Count': [((nb_pred != y_test_arr) & (y_test_arr == 1) & high_conf_mask).sum(),
                 ((nb_pred != y_test_arr) & (y_test_arr == 1) & medium_conf_mask).sum(),
                 ((nb_pred != y_test_arr) & (y_test_arr == 1) & low_conf_mask).sum()],
})
print("\n" + "=" * 65)
print("表7-5 基于置信度的分层决策机制统计")
print("=" * 65)
print(tiered_df.to_string(index=False))

# ==========================================
# 第8章 综合结论表
# ==========================================
print("\n" + "=" * 65)
print("表8-1 模型综合评价总结")
print("=" * 65)

summary_data = []
for name, r in results.items():
    uncertain_r = (np.abs(r['y_prob'] - 0.5) < 0.2).mean()
    avg_entropy = np.mean(-r['y_prob'] * np.log2(r['y_prob'] + 1e-10) -
                          (1 - r['y_prob']) * np.log2(1 - r['y_prob'] + 1e-10))
    summary_data.append({
        'Model': name,
        'Accuracy': f"{r['accuracy']:.4f}",
        'F1-Score': f"{r['f1_score']:.4f}",
        'AUC': f"{r['auc']:.4f}",
        'Log Loss': f"{r['log_loss']:.4f}",
        'Avg Entropy': f"{avg_entropy:.4f}",
        'Uncertain Ratio': f"{uncertain_r:.4f}",
        'Interpretability': 'High' if 'NB' in name else ('Medium' if name == 'Logistic Regression' else 'Low')
    })

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))

# ==========================================
# 图8-1 综合雷达图
# ==========================================
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

categories_radar = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
N = len(categories_radar)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

for idx, (name, r) in enumerate(list(results.items())[:4]):  # 只显示前4个模型
    values = [r['accuracy'], r['precision'], r['recall'], r['f1_score'], r['auc']]
    values += values[:1]
    ax.plot(angles, values, 'o-', color=colors_roc[idx], lw=2, markersize=6, label=name)
    ax.fill(angles, values, alpha=0.1, color=colors_roc[idx])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories_radar, fontsize=12)
ax.set_ylim(0.5, 1.05)
ax.set_title('Figure 8-1: Comprehensive Performance Radar Chart', fontsize=14, y=1.08)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)

plt.tight_layout()
if SAVE_FIG:
    plt.savefig('./fig8_1_radar_chart.png', bbox_inches='tight')
plt.show()
print("[图8-1] 综合雷达图已生成")

# ==========================================
# 导出所有表格到Excel
# ==========================================
with pd.ExcelWriter('./spam_classification_results.xlsx', engine='openpyxl') as writer:
    data_info.to_excel(writer, sheet_name='数据集信息', index=False)
    metrics_df.to_excel(writer, sheet_name='模型性能对比')
    prob_stats_df.to_excel(writer, sheet_name='概率区间统计', index=False)
    conf_df.to_excel(writer, sheet_name='置信度统计', index=False)
    smooth_df.to_excel(writer, sheet_name='平滑参数影响', index=False)
    tiered_df.to_excel(writer, sheet_name='分层决策统计', index=False)
    summary_df.to_excel(writer, sheet_name='综合评价', index=False)

print("\n" + "=" * 65)
print("所有实验结果已保存！")
print("=" * 65)
print("生成的图片文件:")
print("  fig2_1_class_distribution.png     - 类别分布图")
print("  fig6_1_performance_comparison.png - 性能对比柱状图")
print("  fig6_2_confusion_matrices.png     - 混淆矩阵")
print("  fig6_3_roc_curves.png             - ROC曲线")
print("  fig6_4_pr_curves.png              - PR曲线")
print("  fig7_1_probability_distribution.png - 概率分布直方图")
print("  fig7_2_confidence_analysis.png    - 置信度分析")
print("  fig7_3_uncertain_analysis.png     - 不确定样本分析")
print("  fig7_4_calibration.png            - 概率校准曲线")
print("  fig7_5_uncertainty_entropy_risk.png - 熵与决策风险")
print("  fig7_6_cv_stability.png           - 交叉验证与学习曲线")
print("  fig7_7_smoothing_analysis.png     - 平滑参数分析")
print("  fig7_8_feature_importance.png     - 特征重要性")
print("  fig7_9_tiered_decision.png        - 分层决策机制")
print("  fig8_1_radar_chart.png            - 综合雷达图")
print("\n生成的表格文件:")
print("  spam_classification_results.xlsx  - 包含所有结果表格")

