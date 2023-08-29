#!/usr/bin/env python
"""
This script splits the provided dataframe in test and remainder
"""
import argparse
import logging
import tempfile
import pandas as pd
import wandb
from sklearn.model_selection import train_test_split
from wandb_utils.log_artifact import log_artifact

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def segregate_data(args):
    """
    Segregate data into train and test
    """

    run = wandb.init(job_type="train_val_test_split")
    run.config.update(args)

    # Download input artifact. This will also note that this script is using this
    # particular version of the artifact
    logger.info("Fetching artifact %s", args.input)
    artifact_local_path = run.use_artifact(args.input).file()

    data_frame = pd.read_csv(artifact_local_path)

    logger.info("Splitting trainval and test")
    trainval, test = train_test_split(
        data_frame,
        test_size=args.test_size,
        random_state=args.random_seed,
        stratify=data_frame[args.stratify_by] if args.stratify_by != 'none' else None,
    )

    # Save to output files
    for data_frame, k in zip([trainval, test], ['trainval', 'test']):
        logger.info("Uploading %s_data.csv dataset", k)
        with tempfile.NamedTemporaryFile("w") as file:

            data_frame.to_csv(file.name, index=False)

            log_artifact(
                "{k}_data.csv".format(k=k),
                "{k}_data".format(k=k),
                "{k} split of dataset".format(k=k),
                file.name,
                run,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split test and remainder")

    parser.add_argument("input", type=str, help="Input artifact to split")

    parser.add_argument(
        "test_size",
        type=float,
        help="Size of the test split. Fraction of the dataset, or number of items")

    parser.add_argument(
        "--random_seed",
        type=int,
        help="Seed for random number generator",
        default=42,
        required=False)

    parser.add_argument(
        "--stratify_by",
        type=str,
        help="Column to use for stratification",
        default='none',
        required=False)

    args = parser.parse_args()

    segregate_data(args)
