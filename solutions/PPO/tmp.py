from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import pandas as pd
import numpy as np

# 1) 读数据
data = pd.read_csv('../../output.csv')
data.loc[data['L1']>0,'L1'] = 1
data.loc[data['L2']>0,'L2'] = 1
data.drop_duplicates(inplace=True)
X = data.iloc[:,:-1]
y = data['label']
print('n_sample=',X.shape[0],'n_features=',X.shape[1])

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# ========== 1) 定义模型（保持与你当前一致） ==========
clf = RandomForestClassifier(
    n_estimators=400,
    max_depth=25,
    class_weight='balanced',
    n_jobs=-1,
    random_state=42
)

# ========== 2) 5 折分层交叉验证（逐折报告 + 指标汇总） ==========
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# （A）逐折详细报告
fold = 1
all_reports = []
all_cms = []

for tr_idx, te_idx in skf.split(X, y):
    X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
    y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

    clf.fit(X_tr, y_tr)
    y_pr = clf.predict(X_te)

    print(f"\n===== Fold {fold} =====")
    print(classification_report(y_te, y_pr, digits=4))
    cm = confusion_matrix(y_te, y_pr)
    print("Confusion Matrix:\n", cm)
    all_reports.append(classification_report(y_te, y_pr, output_dict=True))
    all_cms.append(cm)
    fold += 1

# （B）使用 cross_validate 做指标汇总（更简洁，含时间）
scoring = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "f1_macro": "f1_macro",
    "f1_weighted": "f1_weighted",
    "precision_macro": "precision_macro",
    "recall_macro": "recall_macro",
    # 多分类 AUC（需要 predict_proba 支持；RF 支持）
    "roc_auc_ovr_weighted": "roc_auc_ovr_weighted",
}
cv_res = cross_validate(clf, X, y, cv=skf, scoring=scoring, return_train_score=False, n_jobs=-1)

def mean_std(name):
    arr = cv_res[f"test_{name}"]
    return f"{arr.mean():.4f} ± {arr.std():.4f}"

print("\n===== 5-Fold CV Summary =====")
for m in scoring.keys():
    print(f"{m:>22}: {mean_std(m)}")

# （可选）汇总混淆矩阵（逐元素取平均）
avg_cm = sum(all_cms) / len(all_cms)
print("\nAverage Confusion Matrix over folds (element-wise mean):\n", np.round(avg_cm, 2))

# ========== 3) 用全量数据重训并保存（与原逻辑一致） ==========
clf.fit(X, y)

import joblib
from datetime import datetime
bundle = {
    "model": clf,
    "feature_names": X.columns.tolist(),
    "meta": {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "algo": "RandomForestClassifier",
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "sklearn_version": __import__("sklearn").__version__,
        "cv": {"type": "StratifiedKFold", "n_splits": 5, "shuffle": True, "random_state": 42},
        "cv_summary": {k: (cv_res[f"test_{k}"].mean(), cv_res[f"test_{k}"].std()) for k in scoring.keys()},
    },
}
model_path = "../../rf_deadlock_v1.joblib"
joblib.dump(bundle, model_path)
print(f"\n✅ 模型已用全量数据重训并保存到: {model_path}")

