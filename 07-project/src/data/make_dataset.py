# -*- coding: utf-8 -*-
import logging
from pathlib import Path

import click
from dotenv import find_dotenv, load_dotenv
import polars as pl
import polars.selectors as cs

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
    with open(output_path, 'w') as f:
        df.write_csv(f)
    

class Preprocessor:
    def __init__(self, data):
        self.data = data

    def shuffle_data(self):
        ids_shuffled = self.data.select(pl.col('Id').shuffle(seed=42))
        self.data = self.data.join(ids_shuffled, how = 'inner', on = 'Id')
        return self

    def split_data(self):
        data_rows = self.data.shape[0]
        self.data_train = self.data.slice(0, int(data_rows * 0.7))
        self.data_val = self.data.slice(int(data_rows * 0.7), int(data_rows * 0.15))
        self.data_test = self.data.slice(int(data_rows * 0.85), data_rows)
        return self

    def split_X_y(self):
        self.X_train = self.data_train.drop('quality', 'Id')
        self.y_train = self.data_train.select('quality')

        self.X_val = self.data_val.drop('quality', 'Id')
        self.y_val = self.data_val.select('quality')

        self.X_test = self.data_test.drop('quality', 'Id')
        self.y_test = self.data_test.select('quality')

        return self

    def standardize_data(self):
        """
        This is done to make sure that the data is centered around 0 and has a standard deviation of 1
        Polars does not have a built in standardization function, so we have to do it manually
        This is done separately for each dataset to avoid data leakage        
        """
        for dataset in [self.X_train, self.X_val, self.X_test]:
            for col_name in dataset.columns:
                col_mean = dataset[col_name].mean()
                col_std = dataset[col_name].std()
                dataset = dataset.with_columns(((dataset[col_name] - col_mean) / col_std).alias(f'{col_name}_std'))
            dataset = dataset.lazy().select(cs.ends_with('std')).collect()

        return self


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    
    # Import data
    logger.info(f'input filepath is {input_filepath}')
    data = import_data(df_path = Path(input_filepath, 'WineQT.csv').resolve())

    # Run Preprocessing
    preprocessor = Preprocessor(data)
    preprocessor.shuffle_data().split_data().split_X_y().standardize_data()
    logger.info('Initialized Preprocessor')

    X_train = preprocessor.X_train
    y_train = preprocessor.y_train

    X_val = preprocessor.X_val
    y_val = preprocessor.y_val

    X_test = preprocessor.X_test
    y_test = preprocessor.y_test
    logger.info('Finished Preprocessing')
    
    data_dict = {
        'X_train': X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test
    }

    logger.info('Saving data to csv')
    for dataset_name, dataset in data_dict.items():
        export_data(output_path = Path(output_filepath, f'{dataset_name}.csv'), df = dataset)
    logger.info('Finished saving data to csv')

    return X_train, y_train, X_val, y_val, X_test, y_test
    

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
