import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score



# ==========================
# 1. 特征工程函数
# ==========================

def feature_engineering(data):

    data = data.copy()


    # ----------------------
    # Title
    # ----------------------

    data["Title"] = data["Name"].str.extract(
        ' ([A-Za-z]+)\.',
        expand=False
    )


    data["Title"] = data["Title"].replace(
        {
            "Mme": "Mrs",
            "Ms": "Miss",
            "Mlle": "Miss"
        }
    )


    rare_titles = [
        "Dr",
        "Rev",
        "Col",
        "Major",
        "Countess",
        "Sir",
        "Lady",
        "Jonkheer",
        "Don",
        "Capt"
    ]


    data["Title"] = data["Title"].replace(
        rare_titles,
        "Rare"
    )



    # ----------------------
    # Family Features
    # ----------------------

    data["FamilySize"] = (
        data["SibSp"]
        +
        data["Parch"]
        +
        1
    )


    data["IsAlone"] = 0

    data.loc[
        data["FamilySize"] == 1,
        "IsAlone"
    ] = 1



    data["FamilyGroup"] = pd.cut(
        data["FamilySize"],
        bins=[0,1,4,7,20],
        labels=[
            "Single",
            "Small",
            "Medium",
            "Large"
        ]
    )



    # ----------------------
    # Fare处理
    # 注意：
    # 必须先填充，再qcut
    # ----------------------

    data["Fare"] = data["Fare"].fillna(
        data["Fare"].median()
    )


    data["FareGroup"] = pd.qcut(
        data["Fare"],
        4,
        labels=False,
        duplicates="drop"
    )



    # ----------------------
    # Age处理
    # ----------------------

    data["Age"] = data.groupby(
        "Title"
    )["Age"].transform(
        lambda x: x.fillna(x.median())
    )


    # 防止某些Title仍然没有年龄

    data["Age"] = data["Age"].fillna(
        data["Age"].median()
    )



    # ----------------------
    # Embarked
    # ----------------------

    data["Embarked"] = data["Embarked"].fillna(
        data["Embarked"].mode()[0]
    )



    # ----------------------
    # 删除无用列
    # ----------------------

    data = data.drop(
        [
            "PassengerId",
            "Name",
            "Ticket",
            "Cabin"
        ],
        axis=1
    )



    # ----------------------
    # Sex编码
    # ----------------------

    data["Sex"] = data["Sex"].map(
        {
            "male":0,
            "female":1
        }
    )



    # ----------------------
    # One-Hot
    # ----------------------

    data = pd.get_dummies(
        data,
        columns=[
            "Title",
            "Embarked",
            "FamilyGroup"
        ]
    )


    return data





# ==========================
# 2. 读取训练集
# ==========================

train = pd.read_csv(
    "train.csv"
)



train2 = feature_engineering(
    train
)



X = train2.drop(
    "Survived",
    axis=1
)


y = train2["Survived"]





# ==========================
# 3. 建立模型
# ==========================

model = GradientBoostingClassifier(

    n_estimators=100,

    learning_rate=0.05,

    max_depth=3,

    random_state=42
)





# ==========================
# 4. 五折交叉验证
# ==========================

scores = cross_val_score(
    model,
    X,
    y,
    cv=5
)



print("CV结果:")
print(scores)


print(
    "CV平均:",
    scores.mean()
)





# ==========================
# 5. 使用全部训练数据训练
# ==========================

model.fit(
    X,
    y
)





# ==========================
# 6. 处理测试集
# ==========================

test = pd.read_csv(
    "test.csv"
)


test_id = test["PassengerId"]



test2 = feature_engineering(
    test
)



# 检查NaN

print(
    "测试集NaN数量:",
    test2.isnull().sum().sum()
)



# 保持训练测试列一致

test2 = test2.reindex(
    columns=X.columns,
    fill_value=0
)





# ==========================
# 7. 预测
# ==========================

prediction = model.predict(
    test2
)





# ==========================
# 8. 生成提交文件
# ==========================

submission = pd.DataFrame(
    {
        "PassengerId": test_id,
        "Survived": prediction
    }
)



submission.to_csv(
    "submission.csv",
    index=False
)



print(
    "submission.csv生成成功"
)


print(
    submission.head()
)