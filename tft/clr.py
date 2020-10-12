import pandas as pd
import os
#from script_download_data import main as download_data
from libs.tft_model import TemporalFusionTransformer
from data_formatters.base import GenericDataFormatter, DataTypes, InputTypes
from libs import utils        # Load TFT helper functions
import sklearn.preprocessing  # Used for data standardization
from data_formatters.clr_store import StoreFormatter
#import os
import tensorflow as tf
import logging
from datetime import date

logging.basicConfig(level=logging.DEBUG, filename="clr_log.txt")
USE_GPU = False


def get_data():

    output_folder = os.path.join(os.getcwd(), 'outputs')
    force_download = False  # Skips download if data is already present

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    converters = {
        # "receipt_date" :date,
        "district_id": str,
        "district_name": str,
        "store_id": str,
        "store_name": str,
        "department_id": str,
        "department_id_int": str,
        "department_description": str,
        "category_id_int": str,
        "category_description": str,
        "sub_category_id_int": str,
        "sub_category_description": str,
        "segment_id_int": str,
        "segment_description": str,
        "article_id": str,
        "article_quantity": int,
        "article_gst_excluded_amount": float
    }
    '''converters = {
       0 :date,
        1: str,
        2: str,
        3: str,
        4:str,
        5:str,
        6:int,
        7:str,
        8: int,
        9:str,
        10: int,
        11:str,
        12: int,
        13: str,
        14: str,
        15:int,
        16:float        
    }'''
    csv_file = os.path.join(output_folder, 'data',
                            'clr', 'ponsonby.csv')

    # Load the downloaded data
    df = pd.read_csv(csv_file, index_col=False, converters=converters,
                     na_filter=False, parse_dates=['receipt_date'])
    logging.info('columns')
    logging.info(df.columns)
    row = df.iloc[0]
    logging.info(row)

    return df, output_folder


def run_train():
    if USE_GPU:
        tf_config = utils.get_default_tensorflow_config(
            tf_device="gpu", gpu_id=0)
    else:
        tf_config = utils.get_default_tensorflow_config(
            tf_device="cpu", gpu_id=0)

    tf.compat.v1.reset_default_graph()
    with tf.Graph().as_default(), tf.compat.v1.Session(config=tf_config) as sess:
        # name tf.keras.backend.set_session is deprecated. Please use tf.compat.v1.keras.backend.set_sessio
        tf.compat.v1.keras.backend.set_session(sess)

        # Create a TFT model
        model = TemporalFusionTransformer(model_params,
                                          use_cudnn=USE_GPU)  # Don't Run model on GPU using CuDNNLSTM cells

        # Sample data into minibatches for training
        if not model.training_data_cached():
            model.cache_batched_data(train, "train", num_samples=45000)
            model.cache_batched_data(valid, "valid", num_samples=5000)

        # Train and save model
        model.fit()
        model.save(model_folder)
    return model


def predict():
    tf.compat.v1.reset_default_graph()
    with tf.Graph().as_default(), tf.Session(config=tf_config) as sess:

        tf.compat.v1.keras.backend.set_session(sess)

        # Create a new model & load weights
        model = TemporalFusionTransformer(model_params,
                                          use_cudnn=True)
        model.load(model_folder)

        # Make forecasts
        logging.info("Starting predictions")

        output_map = model.predict(test, return_targets=True)

        targets = data_formatter.format_predictions(output_map["targets"])

        # Format predictions
        p50_forecast = data_formatter.format_predictions(output_map["p50"])
        p90_forecast = data_formatter.format_predictions(output_map["p90"])

        def extract_numerical_data(data):
            """Strips out forecast time and identifier columns."""
            return data[[
                col for col in data.columns
                if col not in {"forecast_time", "identifier"}
            ]]

        # Compute Losses
        p50_loss = utils.numpy_normalised_quantile_loss(
            extract_numerical_data(
                targets), extract_numerical_data(p50_forecast),
            0.5)
        p90_loss = utils.numpy_normalised_quantile_loss(
            extract_numerical_data(
                targets), extract_numerical_data(p90_forecast),
            0.9)
    print("Normalised quantile losses: P50={}, P90={}".format(
        p50_loss.mean(), p90_loss.mean()))


if __name__ == '__main__':
    logging.info("Starting")
    df, output_folder = get_data()
    # Create a data formatter
    data_formatter = StoreFormatter()

    # Split data

    logging.info("Splitting data")
    train, valid, test = data_formatter.split_data(df)

    logging.info("getting experiment params")
    data_params = data_formatter.get_experiment_params()

    # Model parameters for calibration
    model_params = {'dropout_rate': 0.3,     # Dropout discard rate
                    'hidden_layer_size': 320,  # Internal state size of TFT
                    'learning_rate': 0.001,   # ADAM initial learning rate
                    'minibatch_size': 128,    # Minibatch size for training
                    'max_gradient_norm': 100.,  # Max norm for gradient clipping
                    'num_heads': 4,           # Number of heads for multi-head attention
                    # Number of stacks (default 1 for interpretability)
                    'stack_size': 1
                    }

    # Folder to save network weights during training.
    model_folder = os.path.join(
        output_folder, 'saved_models', 'traffic', 'fixed')
    model_params['model_folder'] = model_folder

    model_params.update(data_params)
    # Specify GPU usage
    if USE_GPU:
        tf_config = utils.get_default_tensorflow_config(
            tf_device="gpu", gpu_id=0)
    else:
        tf_config = utils.get_default_tensorflow_config(
            tf_device="cpu", gpu_id=0)
    logging.info("Starting Training")

    model = run_train()
    predict()
