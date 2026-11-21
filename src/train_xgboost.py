import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# 1. CSVファイルの読み込み
df = pd.read_csv("data/kasama_merged.csv")

# 2. 特徴量とラベルに分割
features = ["NDVI_flood", "NDVI_grow", "VH_min", "VH_winter"]
X = df[features]

le = LabelEncoder()
y = le.fit_transform(df["label"])

# 3. 学習データとテストデータに分割（両モデルで完全に同一の分割を使用）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 4. 不均衡補正パラメータの計算
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

# 5. モデルの定義
models = {
    "XGBoost": XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        max_depth=3,
        learning_rate=0.05,
        n_estimators=100,
        random_state=42
    ),
    "LightGBM": LGBMClassifier(
        scale_pos_weight=scale_pos_weight,
        max_depth=3,
        num_leaves=7,          # max_depth=3 に合わせた葉の上限 (2^3 - 1)
        min_child_samples=10,  # 少数データ向けにデフォルト(20)から引き下げ
        learning_rate=0.05,
        n_estimators=100,
        random_state=42,
        verbose=-1             # 学習時のログ出力を抑制
    )
}

# 6. 各モデルで学習・予測・評価
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(f"================ {name} ================")
    print("【混同行列 (Confusion Matrix)】")
    print(confusion_matrix(y_test, y_pred))
    print("\n【分類レポート (Classification Report)】")
    print(classification_report(y_test, y_pred, target_names=le.classes_.astype(str)))
    print("【実際のB(7件)に対する予測確率】")
    print(y_proba[y_test == 1].round(3))
    print()