import pandas as pd

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


