import os
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

RANDOM_STATE = 42

df = pd.read_csv("metadata.csv")

print("Total images:", len(df))
print("Unique patients:", df["patient_id"].nunique())

# 70% train, 30% temporary
gss1 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.30,
    random_state=RANDOM_STATE
)

train_idx, temp_idx = next(
    gss1.split(
        df,
        y=df["label"],
        groups=df["patient_id"]
    )
)

train_df = df.iloc[train_idx].copy()
temp_df = df.iloc[temp_idx].copy()

# Split remaining 30% equally:
# 15% validation, 15% test
gss2 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.50,
    random_state=RANDOM_STATE
)

val_idx, test_idx = next(
    gss2.split(
        temp_df,
        y=temp_df["label"],
        groups=temp_df["patient_id"]
    )
)

val_df = temp_df.iloc[val_idx].copy()
test_df = temp_df.iloc[test_idx].copy()

# Save splits
train_df.to_csv("train.csv", index=False)
val_df.to_csv("val.csv", index=False)
test_df.to_csv("test.csv", index=False)

print("\nSplit complete!")
print("Train:", len(train_df), "images,", train_df["patient_id"].nunique(), "patients")
print("Val:  ", len(val_df), "images,", val_df["patient_id"].nunique(), "patients")
print("Test: ", len(test_df), "images,", test_df["patient_id"].nunique(), "patients")

# Verify no patient overlap
train_patients = set(train_df["patient_id"])
val_patients = set(val_df["patient_id"])
test_patients = set(test_df["patient_id"])

print("\nPatient overlap:")
print("Train ∩ Val :", len(train_patients & val_patients))
print("Train ∩ Test:", len(train_patients & test_patients))
print("Val ∩ Test  :", len(val_patients & test_patients))

print("\nClass distribution:")
print("\nTrain:")
print(train_df["class_name"].value_counts())

print("\nValidation:")
print(val_df["class_name"].value_counts())

print("\nTest:")
print(test_df["class_name"].value_counts())
