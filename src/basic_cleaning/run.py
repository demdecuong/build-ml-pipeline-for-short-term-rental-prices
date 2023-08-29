#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning,
exporting the result to a new artifact
"""
import argparse
import logging
import os

import pandas as pd
import wandb

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def basic_cleaning(args):
    """
    Function to download data from W&B, apply basic data cleaning,
    and logging

    argument:
        args : command line argument to specify artifact information and
            basic cleaning configuration
    return:
        None
    """

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    logger.info("Downloading artifact")
    artifact = run.use_artifact(args.input_artifact)
    artifact_path = artifact.file()

    # Read artifact
    data_frame = pd.read_csv(artifact_path)

    # Drop outliers
    logger.info("Dropping outlier")
    min_price = args.min_price
    max_price = args.max_price
    idx = data_frame['price'].between(min_price, max_price)
    data_frame = data_frame[idx].copy()

    # Convert last_review to datetime
    logger.info("Convert last_review columns to date")
    data_frame['last_review'] = pd.to_datetime(data_frame['last_review'])

    # Drop Outlier from longitude and latitude
    idx = data_frame['longitude'].between(-74.25, - \
                                          73.50) & data_frame['latitude'].between(40.5, 41.2)
    data_frame = data_frame[idx].copy()

    # Saving the artifact
    filename = args.output_artifact
    data_frame.to_csv(filename, index=False)

    artifact = wandb.Artifact(
        name=args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file(filename)

    logger.info("Logging artifact")
    run.log_artifact(artifact)

    os.remove(filename)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Fully-qualified name for the input artifact",
        required=True
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Fully-qualified name for the input artifact",
        required=True
    )

    parser.add_argument(
        "--output_type",
        type=str,
        help="Type for the artifact",
        required=True
    )

    parser.add_argument(
        "--output_description",
        type=str,
        help="Description for the artifact",
        required=True
    )

    parser.add_argument(
        "--min_price",
        type=float,
        help="Minimum value for price",
        required=True
    )

    parser.add_argument(
        "--max_price",
        type=float,
        help="Maximum value for price",
        required=True
    )

    args = parser.parse_args()

    basic_cleaning(args)
