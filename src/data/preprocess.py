import pandas as pd
from sklearn.preprocessing import LabelEncoder


def preprocess():
    df = pd.read_csv("data/raw/titanic.csv")

    # Drop unnecessary columns
    df.drop(columns=["Cabin", "Name", "Ticket"], inplace=True)

    # Handle missing values
    df["Age"].fillna(df["Age"].mean(), inplace=True)
    df["Embarked"].fillna("S", inplace=True)

    # Encode categorical features
    label_encoder = LabelEncoder()
    df["Sex"] = label_encoder.fit_transform(df["Sex"])
    df["Embarked"] = label_encoder.fit_transform(df["Embarked"])

    # Ensure the order of columns is consistent
    df = df[["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "Embarked",
             "Survived"]]

    # Save the processed dataframe
    df.to_csv("data/processed/titanicp.csv", index=False)


if __name__ == "__main__":
    preprocess()
