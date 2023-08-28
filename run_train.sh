mlflow run https://github.com/demdecuong/build-ml-pipeline-for-short-term-rental-prices.git \
             -v 1.0.1 \
             -P hydra_options="etl.sample='sample2.csv'"