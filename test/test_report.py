from eep.report import check_if_dataset_on_hf, get_cv_results
import subprocess
import os
from pytest import fixture


# fixture for regr_dataset
# need 1hot encoding for gpmodel and unirep_1900 for random forest
@fixture(scope="module")
def needed_encodings():
    return ["aa_1hot", "aa_unirep_1900"]


@fixture(scope="module")
def hf_dataset_name():
    return "Company/CrHydA1_PE_REGR"


def test_check_if_dataset_on_hf(hf_dataset_name):
    # check if the function finds one of the datasets
    assert check_if_dataset_on_hf(hf_dataset_name)


def test_check_if_fake_dataset_on_hf():
    # check if the function does not find a dataset that is not there
    assert not check_if_dataset_on_hf("Company/CrHydA1_regr_fake")


def test_get_cv_results(regr_dataset):
    # check if the function test_get_cv_results runs
    get_cv_results(
        5,
        regr_dataset,
        0.8,
    )


def test_get_report():
    # check if a report can be generated
    command = "python src/eep/report.py Company/CrHydA1_PE_REGR --dataname for_pytest"
    subprocess.run(command, shell=True, check=True)
    # os.remove("Report_for_pytest.html")
    os.remove("for_pytest_cv5.pkl")
    # try:
    #     # Execute the command
    #     subprocess.run(command, shell=True, check=True)
    # except subprocess.CalledProcessError as e:
    #     # Raise a custom exception if the command fails
    #     raise Exception(f"Error executing the command: {e}")
    # except Exception as e:
    #     # Handle any other exceptions here
    #     raise Exception(f"Error executing the command: {e}")
