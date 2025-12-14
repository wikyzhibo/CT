import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
import joblib
from datetime import datetime

# 1) 读数据
data = pd.read_csv('../../output.csv')
data.loc[data['L1']>0,'L1'] = 1
data.loc[data['L2']>0,'L2'] = 1
data.drop_duplicates(inplace=True)
X = data.iloc[:,:-1]
y = data['label']
print('n_sample=',X.shape[0],'n_features=',X.shape[1])

# 2) 定义 Pipeline：标准化 + 深层感知机
mlp = MLPClassifier(
    hidden_layer_sizes=(256, 256, 256),  # 深层结构
    activation='relu',
    solver='adam',
    alpha=1e-4,              # L2 正则
    batch_size=256,
    learning_rate_init=1e-3,
    max_iter=200,
    early_stopping=True,     # 提前停止
    n_iter_no_change=15,
    validation_fraction=0.1, # 从训练集中切一部分做早停验证
    random_state=42
)
pipe = Pipeline([
    ('scaler', StandardScaler(with_mean=True, with_std=True)),
    ('mlp', mlp)
])

# 3) 5 折分层交叉验证（逐折打印 + 汇总）
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("===== 5-Fold Stratified CV (MLP) =====")
all_cms = []
fold = 1
for tr_idx, te_idx in skf.split(X, y):
    X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
    y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

    # 用样本权重处理类不平衡（MLP 不支持 class_weight，但 fit 支持 sample_weight）
    sw = compute_sample_weight(class_weight='balanced', y=y_tr)

    pipe.fit(X_tr, y_tr, mlp__sample_weight=sw)

    y_pr = pipe.predict(X_te)
    print(f"\n--- Fold {fold} ---")
    print(classification_report(y_te, y_pr, digits=4))
    cm = confusion_matrix(y_te, y_pr)
    print("Confusion Matrix:\n", cm)
    all_cms.append(cm)
    fold += 1

# 也可一次性做指标汇总（需要 predict_proba 才能做 ROC-AUC）
scoring = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "f1_macro": "f1_macro",
    "f1_weighted": "f1_weighted",
    "precision_macro": "precision_macro",
    "recall_macro": "recall_macro",
}
cv_res = cross_validate(
    pipe, X, y, cv=skf, scoring=scoring, return_train_score=False, n_jobs=-1,
    fit_params={"mlp__sample_weight": compute_sample_weight('balanced', y=y)}  # 注意：这里用全量 y 估权重做近似
)

def mean_std(arr): return f"{arr.mean():.4f} ± {arr.std():.4f}"
print("\n===== CV Summary =====")
for k in scoring:
    arr = cv_res[f"test_{k}"]
    print(f"{k:>20}: {mean_std(arr)}")

avg_cm = sum(all_cms) / len(all_cms)
print("\nAverage Confusion Matrix over folds:\n", np.round(avg_cm, 2))

# 4) 用全量数据重训并保存（线上/控制器用）
sw_full = compute_sample_weight('balanced', y=y)
pipe.fit(X, y, mlp__sample_weight=sw_full)

bundle = {
    "model": pipe,                              # 包含 scaler + mlp
    "feature_names": X.columns.tolist(),        # 固化特征顺序
    "meta": {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "algo": "MLPClassifier",
        "hidden_layers": (256, 128, 64),
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "sklearn_version": __import__("sklearn").__version__,
        "cv": {"type": "StratifiedKFold", "n_splits": 5, "shuffle": True, "random_state": 42},
        "cv_summary": {k: (float(cv_res[f"test_{k}"].mean()),
                           float(cv_res[f"test_{k}"].std())) for k in scoring},
    },
}
model_path = "mlp_deadlock_v1.joblib"
joblib.dump(bundle, model_path)
print(f"\n✅ MLP 模型已保存到: {model_path}")
