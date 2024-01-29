from eep.models.logistic_regr import LogisticRegr
from pedata.static.example_data.example_data import ClassficationToyDataset

print(
    "===========================================Load dataset==========================================="
)
dataset = ClassficationToyDataset(["aa_1gram", "aa_unirep_1900"]).train_test_split_dataset
train_df = dataset["train"]
test_df = dataset["test"]

print(
    "===========================================Fit and Predict==========================================="
)
print(train_df["aa_1gram"].shape)
model = LogisticRegr()
model.fit(None, train_df)
pred = model.predict(test_df)
print(pred)

print(
    "===========================================Save and Load==========================================="
)
model = LogisticRegr()
model.fit(None, train_df)
pred_0 = model.predict(test_df)
print("pred_0", pred_0)
model.save_weights("logistic_regr")
model = LogisticRegr()
model.load_weights("logistic_regr")
pred_1 = model.predict(test_df)
print("pred_1", pred_1)
assert (pred_0["target high_low"] == pred_1["target high_low"]).all()
