import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 加载数据
train = pd.read_csv("train.csv")

# 创建副本
data2 = train.copy()

# 提取Title特征
data2["Title"] = data2["Name"].str.extract(' ([A-Za-z]+)\.', expand=False)

# Title映射和合并
data2["Title"] = data2["Title"].replace({
    "Mme": "Mrs",
    "Ms": "Miss"
})

rare_titles = [
    "Dr", "Rev", "Col", "Major", "Mlle", 
    "Countess", "Sir", "Lady", "Jonkheer", "Don", "Capt"
]
data2["Title"] = data2["Title"].replace(rare_titles, "Rare")

# 创建FamilySize特征
data2["FamilySize"] = data2["SibSp"] + data2["Parch"] + 1

data2 = data2.drop(
    ["PassengerId", "Name", "Ticket", "Cabin"],
    axis=1
)

# 年龄用中位数填充
data2["Age"] = data2["Age"].fillna(data2["Age"].median())

# 登船港口用众数填充
data2["Embarked"] = data2["Embarked"].fillna(data2["Embarked"].mode()[0])

# Sex编码
data2["Sex"] = data2["Sex"].map({"male": 0, "female": 1})

# One-Hot编码Title和Embarked
data2 = pd.get_dummies(data2, columns=["Title"])
data2 = pd.get_dummies(data2, columns=["Embarked"])

X2 = data2.drop("Survived", axis=1)
y2 = data2["Survived"]

X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X2, y2, test_size=0.2, random_state=42
)

# 随机森林（最终最佳模型，准确率83.24%）
rf2 = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf2.fit(X_train2, y_train2)
rf2_pred = rf2.predict(X_test2)

# 评估
accuracy_score(y_test2, rf2_pred)  # 0.8324

# 加载测试集
test = pd.read_csv("test.csv")

test2 = test.copy()

# Title
test2["Title"] = test2["Name"].str.extract(
    ' ([A-Za-z]+)\.*',
    expand=False
)

test2["Title"] = test2["Title"].replace({
    "Mme": "Mrs",
    "Ms": "Miss"
})

rare_titles = [
    "Dr", "Rev", "Col", "Major", "Mlle",
    "Countess", "Sir", "Lady",
    "Jonkheer", "Don", "Capt"
]

test2["Title"] = test2["Title"].replace(
    rare_titles,
    "Rare"
)

# FamilySize
test2["FamilySize"] = (
    test2["SibSp"] +
    test2["Parch"] +
    1
)

# 删除
test2 = test2.drop(
    ["PassengerId", "Name", "Ticket", "Cabin"],
    axis=1
)

# 填充
test2["Age"] = test2["Age"].fillna(
    test2["Age"].median()
)

test2["Fare"] = test2["Fare"].fillna(
    test2["Fare"].median()
)

test2["Embarked"] = test2["Embarked"].fillna(
    test2["Embarked"].mode()[0]
)

# Sex
test2["Sex"] = test2["Sex"].map({
    "male":0,
    "female":1
})

# One-Hot
test2 = pd.get_dummies(
    test2,
    columns=["Title"]
)

test2 = pd.get_dummies(
    test2,
    columns=["Embarked"]
)


# 和训练集保持一致
test2 = test2.reindex(
    columns=X2.columns,
    fill_value=0
)


# 注意：这里直接使用已经训练好的 rf2
test_pred = rf2.predict(test2)


submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": test_pred
})


submission.to_csv(
    "submission.csv",
    index=False
)

print("submission.csv 已生成")
print(submission.head())
