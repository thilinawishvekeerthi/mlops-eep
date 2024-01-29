from eep.models import KernelRegression
import pytest
from pytest import fixture
import numpy as np


# fixture for regr_dataset_train and regr_dataset_test
@fixture(scope="module")
def needed_encodings():
    return ["aa_1hot"]


def test_if_kernel_regression_works(regr_dataset_train, regr_dataset_test):
    KR = KernelRegression(feature_list=["aa_1hot"])
    KR.fit(rng=0, train_ds=regr_dataset_train, target_standardization=True, epochs=10)
    prediction = KR.predict(test_ds=regr_dataset_test)
    target_name, _ = KR.get_target(regr_dataset_test)
    all(
        prediction[target_name[0]]["mean"].round(2)
        == np.array([2467.243, 2491.269, 2475.938, 2483.098, 2469.530]).round(2)
    )


def test_if_kernel_regression_returns_error_nofeature():
    with pytest.raises(ValueError):
        _ = KernelRegression(feature_list=None)
