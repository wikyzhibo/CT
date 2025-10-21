import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline

# 1) 读数据
dead = pd.read_csv('dead_markings.csv', header=None)
live = pd.read_csv('live_markings.csv', header=None)

# 可选：再去重一次（保险）
dead = dead.drop_duplicates().reset_index(drop=True)
live = live.drop_duplicates().reset_index(drop=True)

# 2) 监督数据集（死=0，活=1 —— 保持你的设定）
X = pd.concat([dead, live], ignore_index=True)
y = np.concatenate([
    np.zeros(len(dead), dtype=np.int64),  # dead -> 0
    np.ones(len(live), dtype=np.int64)    # live -> 1
])

# 3) 划分训练/测试（按类别分层）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y, shuffle=True
)

# 4) 训练随机森林（更稳健的参数）
clf = RandomForestClassifier(
    n_estimators=300,
    class_weight='balanced',
    n_jobs=-1,
    random_state=42
)
clf.fit(X_train, y_train)

# 评估
y_pred = clf.predict(X_test)
y_proba = None
if hasattr(clf, "predict_proba"):
    y_proba = clf.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred, digits=4))
if y_proba is not None:
    print("ROC-AUC:", roc_auc_score(y_test, y_proba))

# 特征重要性（可选：映射列名）
importances = clf.feature_importances_
imp_order = np.argsort(importances)[::-1]
print("\nTop-15 feature importances:")
for i in imp_order[:15]:
    print(f"feat{i:03d}: {importances[i]:.6f}")

# 5) 对 dead 标识做聚类（只聚死锁）
#    用标准化 + KMeans 的 pipeline，k=4 先验，如果想自动选 k 再加轮廓系数搜索
k = 4
kmeans_pipe = make_pipeline(
    StandardScaler(with_mean=True, with_std=True),
    KMeans(n_clusters=k, n_init='auto', random_state=0)
)
kmeans_pipe.fit(dead)

# 聚类标签
dead_labels = kmeans_pipe.named_steps['kmeans'].labels_
dead['cluster'] = dead_labels  # 把簇标签挂回去

# 保存聚类结果（可选）
#dead.to_csv('dead_markings_with_clusters.csv', index=False, header=False)

# 如需把每个簇的“中心”保存出来（注意：中心是在标准化空间的，需要逆变换才回到原尺度）
scaler = kmeans_pipe.named_steps['standardscaler']
kmeans = kmeans_pipe.named_steps['kmeans']
centers_std = kmeans.cluster_centers_
centers = scaler.inverse_transform(centers_std)  # 回到原始标识尺度
#np.savetxt('dead_cluster_centers.csv', centers, delimiter=',')

# 如果你本意是看随机森林对 dead 样本的判别情况（而不是聚类标签），可以单独算：
rf_pred_on_dead = clf.predict(dead.iloc[:, :-1] if 'cluster' in dead.columns else dead)
print("\n[RF] predicted positive-rate on dead set (should be low if labels=dead->0):",
      rf_pred_on_dead.mean())
