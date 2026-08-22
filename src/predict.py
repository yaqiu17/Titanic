import pandas as pd
import joblib

from feature_engineering import feature_engineering



# 加载模型

model_data = joblib.load(
    "../models/model.pkl"
)


model = model_data["model"]

columns = model_data["columns"]



# 测试集

test = pd.read_csv(
    "../data/test.csv"
)


passenger_id = test["PassengerId"]



# 特征处理

test2 = feature_engineering(
    test
)



# 保持列一致

test2 = test2.reindex(
    columns=columns,
    fill_value=0
)



# 预测

prediction = model.predict(
    test2
)



# 保存

submission = pd.DataFrame(
    {
        "PassengerId": passenger_id,
        "Survived": prediction
    }
)



submission.to_csv(
    "../submission/submission.csv",
    index=False
)


print("submission生成成功")