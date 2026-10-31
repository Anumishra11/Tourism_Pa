# for data manipulation
import pandas as pd
import sklearn
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder

DATASET_PATH = "tourism_project/data/tourism.csv"

# A small number of rows contain an unescaped embedded quote in app_name,
# which breaks CSV tokenization; skip just those rows rather than failing.
df = pd.read_csv(DATASET_PATH, on_bad_lines="skip")
print("Dataset loaded successfully.")

# Standardize inconsistent categorical values

# Remove extra spaces and standardize capitalization
df['Gender'] = (
    df['Gender'].astype('string').str.strip().str.title()
)

df['MaritalStatus'] = (
    df['MaritalStatus'].astype('string').str.strip().str.title()
)

# Merge equivalent categories
df['Gender'] = df['Gender'].replace({
    'Fe Male': 'Female'
})

df['MaritalStatus'] = df['MaritalStatus'].replace({
    'Unmarried': 'Single'
});
# Drop unique identifier column (not useful for modeling)
drop_columns = [
    col for col in ['Unnamed: 0', 'CustomerID']
    if col in df.columns
]

df.drop(columns=drop_columns, inplace=True);
print("Columns dropped successfully.");
# Remove post-contact features to prevent data leakage
leakage_columns = [
    'ProductPitched',
    'DurationOfPitch',
    'NumberOfFollowups',
    'PitchSatisfactionScore'
]

# Remove post-contact features to prevent data leakage
leakage_columns = [
    'ProductPitched',
    'DurationOfPitch',
    'NumberOfFollowups',
    'PitchSatisfactionScore'
]

df.drop(
    columns=[
        col for col in leakage_columns if col in df.columns
    ],
    inplace=True
)

# Feature engineering
if 'NumberOfChildrenVisiting' in df.columns:
    df['HasChildren'] = (
        df['NumberOfChildrenVisiting'].fillna(0) > 0
    ).astype(int)

# Define target variable
target_col = 'ProdTaken'

# Split into X (features) and y (target)
X = df.drop(columns=[target_col])
y = df[target_col]

# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Data prepared: train/test splits written.")
