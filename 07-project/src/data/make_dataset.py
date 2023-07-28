# -*- coding: utf-8 -*-
import logging
from datetime import datetime, timedelta
from pathlib import Path

import click
import polars as pl
import polars.selectors as cs
from dotenv import find_dotenv, load_dotenv


def import_data(df_path: Path) -> pl.DataFrame():
    """Import Data

    Args:
        train_df (pathlib.Path): Give path of data

    Returns:
        df (polars.DataFrame): Return Polars Dataframe
    """
    return pl.read_csv(df_path)


def export_data(output_path, df):
    """
    Export Data

    Args:
        output_path (pathlib.Path): Give path of data
        df (polars.DataFrame): Return Polars Dataframe

    Returns:
        None
    """
    logging.info(f"exporting df to {output_path}")
    with open(output_path, "w", encoding="utf8") as f:
        df.write_csv(f)


class Preprocessor:
    # pylint: disable=missing-module-docstring
    """ """

    def __init__(self, data):
        self.data = data

    def shuffle_data(self):
        """ """
        ids_shuffled = self.data.select(pl.col("Id").shuffle(seed=42))
        self.data = self.data.join(ids_shuffled, how="inner", on="Id")
        return self.data

    def split_data(self, data):
        """ """
        data_rows = data.shape[0]
        data_train = data.slice(0, int(data_rows * 0.7))
        data_val = data.slice(int(data_rows * 0.7), int(data_rows * 0.15))
        data_test = data.slice(int(data_rows * 0.85), data_rows)
        return data_train, data_val, data_test

    def split_X_y(self, data):
        """ """
        X = data.drop("quality")
        y = data.select("quality")
        return X, y

    def standardize_data(self, data, id_col="Id"):
        """
        To get to STD 1 and Mean 0
        Polars does not have a built in standardization function, so we have to do it manually
        This is done separately for each dataset to avoid data leakage
        """
        cols_to_grab = data.select(cs.integer().exclude(id_col), cs.float()).columns
        for col_name in cols_to_grab:
            if col_name == id_col:
                None
            else:
                col_mean = data[col_name].mean()
                col_std = data[col_name].std()
                data = data.with_columns(
                    ((data[col_name] - col_mean) / col_std).alias(f"{col_name}_std")
                )
        return data


def create_streaming_date_column(data, starting_date=datetime(2023, 1, 1)):
    # Find the number of rows in the DataFrame
    num_rows = data.shape[0]

    # Generate a list of dates with two of each date
    dates = [starting_date + timedelta(days=i // 2) for i in range(num_rows)]

    return data.with_columns(pl.Series(dates).alias("date"))


def process_and_save_data(data, processor, name, output_filepath):
    """Processes data using the given processor and saves it to a file."""

    X, y = processor.split_X_y(data)
    X = processor.standardize_data(X)

    for dataset_name, dataset in [(f"X_{name}", X), (f"y_{name}", y)]:
        export_data(
            output_path=Path(output_filepath, f"{dataset_name}.csv"), df=dataset
        )


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    # Import data
    logger.info(f"input filepath is {input_filepath}")
    data = import_data(df_path=Path(input_filepath, "WineQT.csv").resolve())

    # Initialize Pre-Processor
    preprocessor = Preprocessor(data)

    # Shuffle the Data
    shuffled_data = preprocessor.shuffle_data()

    # Split
    data_train, data_val, data_test = preprocessor.split_data(shuffled_data)

    # Process and save data
    logger.info("Saving data to csv")
    process_and_save_data(data_train, preprocessor, "train", output_filepath)
    process_and_save_data(data_val, preprocessor, "val", output_filepath)
    data_test = create_streaming_date_column(data_test)
    process_and_save_data(data_test, preprocessor, "test", output_filepath)
    logger.info("Finished saving data to csv")


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    LOG_FMT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=LOG_FMT)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
