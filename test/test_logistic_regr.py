import pytest
from eep.models.logistic_regr import LogisticRegr
from pedata import ClassificationToyDataset
import numpy as np
import tempfile


# ===========  FIXTURES ----------------


@pytest.fixture(scope="module")
def dataset_target_name():
    return "target_high_low"


@pytest.fixture
def dataset():
    return ClassificationToyDataset(
        ["aa_1gram", "aa_unirep_1900"]
    ).train_test_split_dataset


@pytest.fixture
def dataset_0(dataset, dataset_target_name):
    dataset = dataset.remove_columns(dataset_target_name)
    return dataset


@pytest.fixture
def dataset_2(dataset):
    dataset = dataset.map(lambda x: {"target dummy": 1})
    return dataset


@pytest.fixture
def dataset_nonbinary(dataset, dataset_target_name):
    # assigns random target between 0 and 3
    dataset = dataset.map(lambda x: {dataset_target_name: np.random.randint(0, 4)})
    return dataset


# ========== TESTS ---------------------
def test_fit_pred(dataset, dataset_target_name):
    """Test if the pretrained model Pred returns predictions"""
    model = LogisticRegr()
    model.fit(None, dataset["train"])
    pred = model.predict(dataset["test"])
    pred = np.round(pred[dataset_target_name], 8)
    assert pred.shape == (
        len(dataset["test"]),
        len(set(dataset["train"][dataset_target_name])),
    )


def test_save_load(dataset, dataset_target_name):
    """Test if the pretrained model Pred can be saved and loaded"""
    model = LogisticRegr()
    model.fit(None, dataset["train"])
    pred_0 = model.predict(dataset["test"])
    pred_0 = np.round(pred_0[dataset_target_name], 8)
    fp = tempfile.TemporaryFile()
    model.save_weights(fp)
    fp.seek(0)
    model = LogisticRegr()
    model.load_weights(fp)
    pred_1 = model.predict(dataset["test"])
    pred_1 = np.round(pred_1[dataset_target_name], 8)
    assert (pred_0 == pred_1).all()


def test_get_target(dataset_0, dataset_2, dataset_nonbinary):
    """Test if the model raises an error when there are no target variables"""
    model = LogisticRegr()
    with pytest.raises(ValueError):
        model.fit(None, dataset_0["train"])

    with pytest.raises(ValueError):
        model.fit(None, dataset_2["train"])

    with pytest.raises(ValueError):
        model.fit(None, dataset_nonbinary["train"])
