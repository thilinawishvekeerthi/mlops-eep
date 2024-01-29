from eep.models import LowerGrowthTemperaturePred, SolubilityPred
from datasets import load_dataset
from pytest import fixture


# ===========  FIXTURES ----------------
@fixture
def dataset_solubility():
    dataset_name = "Company/Proteinea_solubility_v2"
    return load_dataset(dataset_name, split="test[:2]")


@fixture
def dataset_temperature():
    dataset_name = "Company/TemStaPro-Minor-bal65"
    return load_dataset(dataset_name, split="testing[:2]")


# ========== TESTS ---------------------


def test_SolubilityPred_loading():
    """Test if the pretrained model SolubilityPred loads from gcloud or a local cache"""
    _ = SolubilityPred()


def test_SolubilityPred_predicts(dataset_solubility):
    """Test if the pretrained model SolubilityPred returns predictions"""
    model = SolubilityPred()
    _ = model.predict(test_df=dataset_solubility)


def test_LowerGrowthTemperaturePred_loading():
    """Test if the pretrained model LowerGrowthTemperaturePred loads from gcloud or a local cache"""
    _ = LowerGrowthTemperaturePred()


def test_LowerGrowthTemperaturePred_predicts(dataset_temperature):
    """Test if the pretrained model LowerGrowthTemperaturePred returns predictions"""
    model = LowerGrowthTemperaturePred()
    _ = model.predict(test_df=dataset_temperature)
