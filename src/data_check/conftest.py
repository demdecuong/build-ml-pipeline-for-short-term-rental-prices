"""
Helper function to conduct data testing
"""
import pytest
import pandas as pd
import wandb


def pytest_addoption(parser):
    """
    Add argument option to console
    """
    parser.addoption("--csv", action="store")
    parser.addoption("--ref", action="store")
    parser.addoption("--kl_threshold", action="store")
    parser.addoption("--min_price", action="store")
    parser.addoption("--max_price", action="store")


@pytest.fixture(scope='session')
def data(request):
    """
    Download and read data from W&B artifact
    """
    run = wandb.init(job_type="data_tests", resume=True)

    data_path = run.use_artifact(request.config.option.csv).file()

    if data_path is None:
        pytest.fail("You must provide the --csv option on the command line")

    data_frame = pd.read_csv(data_path)

    return data_frame


@pytest.fixture(scope='session')
def ref_data(request):
    """
    Download and read reference data from W&B artifact
    """
    run = wandb.init(job_type="data_tests", resume=True)

    # Download input artifact. This will also note that this script is using this
    # particular version of the artifact
    data_path = run.use_artifact(request.config.option.ref).file()

    if data_path is None:
        pytest.fail("You must provide the --ref option on the command line")

    data_frame = pd.read_csv(data_path)

    return data_frame


@pytest.fixture(scope='session')
def kl_threshold(request):
    """
    Helper function to specify kl threshold value
    """
    kl_threshold = request.config.option.kl_threshold

    if kl_threshold is None:
        pytest.fail("You must provide a threshold for the KL test")

    return float(kl_threshold)


@pytest.fixture(scope='session')
def min_price(request):
    """
    Helper function to specify min price value
    """
    min_price = request.config.option.min_price

    if min_price is None:
        pytest.fail("You must provide min_price")

    return float(min_price)


@pytest.fixture(scope='session')
def max_price(request):
    """
    Helper function to specify max price value
    """
    max_price = request.config.option.max_price

    if max_price is None:
        pytest.fail("You must provide max_price")

    return float(max_price)
