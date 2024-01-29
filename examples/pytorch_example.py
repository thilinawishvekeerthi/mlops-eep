from eep.models import LowerGrowthTemperaturePred, SolubilityPred
from datasets import load_dataset

model = SolubilityPred(["ankh"])

dataset_name = "Company/Proteinea_solubility_v2"
dataset = load_dataset(dataset_name, split="test[:1%]")
predictions = model.predict(test_df=dataset)
print(predictions)

model = LowerGrowthTemperaturePred(["ankh"])

dataset_name = "Company/TemStaPro-Minor-bal65"
dataset = load_dataset(dataset_name, split="testing[:1%]")
predictions = model.predict(test_df=dataset)
print(predictions)
