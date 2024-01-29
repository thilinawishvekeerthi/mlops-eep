import eep.report as r
import pedata.config.paths as p
import datasets as ds
from datasets import load_dataset
import pandas as pd


if __name__ == "__main__":
    nruns = 10  # number of cross validation runs

    # df = ds.load_from_disk(p.PE_DATA_DIR / "CrHydA1_2022-9-6")

    # loading from huggingface hub
    dataset = load_dataset("Company/CrHydA1_PE_REGR")["whole_dataset"]

    # The following loads the dataset from a CSV file
    # df = dc.preprocess_data("example_dataset.csv")

    # If the dataset should be written directly to AWS S3 buckets after preprocessing/error checking, use the following call
    # df = dc.preprocess_data("example_dataset.csv", S3_BASE_PATH + dataname, OBJECT_OF_TYPE_ds.filesystems.S3FileSystem)

    # load from S3 using
    # df = ds.load_from_disk(S3_BASE_PATH + dataname, OBJECT_OF_TYPE_ds.filesystems.S3FileSystem)

    # compute cross validation results
    cv_runs, avg_perf, good_regressor= r.get_cv_results(nruns, dataset)
    # The following gets the whole self-contained HTML code report as a single string
    rep_html = r.get_report(cv_runs, avg_perf, "Example Dataset")

    with open("Report.html", "w") as f:
        f.write(rep_html)
