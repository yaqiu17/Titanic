import pandas as pd
import joblib

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

from feature_engineering import feature_engineering



# 读取数据

train = pd.read_csv(
    "../data/train.csv"
)



# 特征处理

train2 = feature_engineering(train)



X = train2.drop(
    "Survived",
    axis=1
)


y = train2["Survived"]



# 模型

model = GradientBoostingClassifier(

    n_estimators=100,

    learning_rate=0.05,

    max_depth=3,

    random_state=42
)



# CV

scores = cross_val_score(
    model,
    X,
    y,
    cv=5
)


print(scores)

print(
    "CV平均:",
    scores.mean()
)



# 全量训练

model.fit(
    X,
    y
)



# 保存模型和列名

joblib.dump(
    {
        "model": model,
        "columns": X.columns
    },
    "../models/model.pkl"
)


print("模型保存完成")