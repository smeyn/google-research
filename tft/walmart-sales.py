import sklearn.preprocessing  # Used for data standardization
from libs import utils        # Load TFT helper functions
from data_formatters.base import GenericDataFormatter, DataTypes, InputTypes
from libs.tft_model import TemporalFusionTransformer
from script_download_data import main as download_data
import pandas as pd
import os
#from TrafficFormatter import TrafficFormatter
import os
import tensorflow as tf
import logging
from logging.config import dictConfig
from pandas import Timestamp

USE_GPU = True
#input_path = "/home/jupyter/long_std.feather"
#input_path = "D:\\steph\\Google Drive\\src\\Kaggle\\M5-WALMART\\DATA\\combined.csv"
input_path = "D:\\steph\\Google Drive\\src\\Kaggle\\M5-WALMART\\DATA\\combined.feather"
output_folder = "D:\\steph\\Google Drive\\src\\Kaggle\\M5-WALMART\\output"
logging_config = {
    "version": 1,
    "formatters": {
        "simple": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"}
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "simple",
            "stream": "ext://sys.stdout",
        },
        "infolog": {
            "class": "logging.FileHandler",
            "level": "DEBUG",
            "formatter": "simple",
            "filename": "walmartsales.log",
        },
    },
    "root": {"level": "DEBUG", "handlers": ["console", "infolog"]},
}
dictConfig(logging_config)
LOG = logging.getLogger(__name__)
from data_formatters.walmart import SalesFormatter

 


def run_train():
    LOG.info("Start Training")
    if USE_GPU:
        tf_config = utils.get_default_tensorflow_config(
            tf_device="gpu", gpu_id=0)
    else:
        tf_config = utils.get_default_tensorflow_config(
            tf_device="cpu", gpu_id=0)

    tf.compat.v1.reset_default_graph()
    with tf.Graph().as_default(), tf.compat.v1.Session(config=tf_config) as sess:

        tf.compat.v1.keras.backend.set_session(sess)

        # Create a TFT model
        model = TemporalFusionTransformer(model_params,
                                          use_cudnn=USE_GPU)  # Don't Run model on GPU using CuDNNLSTM cells

        # Sample data into minibatches for training
        if not model.training_data_cached():
            model.cache_batched_data(train, "train", num_samples=450000)
            model.cache_batched_data(valid, "valid", num_samples=50000)

        # Train and save model
        model.fit()
        LOG.info(f"saving trained model to {model_folder}")
        model.save(model_folder)
    return model


if __name__ == '__main__':
    LOG.info("Starting")
    from data_formatters.base import GenericDataFormatter, DataTypes, InputTypes

    df = pd.read_feather(input_path)
    """batch_size=1000
    label_name = 'value'
    select_columns = ['date', 'id', 'value', 'wm_yr_wk', 'weekday', 'wday', 'month', 'year',
       'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2',
       'snap_CA', 'snap_TX', 'snap_WI', 'store_id', 'item_id', 'sell_price',
       'cat_id', 'dept_id', 'state_id'
    ]
    df  = tf.data.experimental.make_csv_dataset(
        input_path, batch_size=batch_size,  
        label_name=label_name, select_columns=select_columns,   shuffle=True,
        shuffle_buffer_size=10000, shuffle_seed=None, prefetch_buffer_size=None,
        num_parallel_reads=None, sloppy=False, num_rows_for_inference=100,
        compression_type=None, ignore_errors=False
        )
    """
    # Create a data formatter
    data_formatter = SalesFormatter()

    # Split data [d_1 - d_1913] train * test [d1914-d1941] for validation
    d1913 = Timestamp('2016-04-25 00:00:00')
    d1850 = Timestamp('2016-02-22 00:00:00')
    LOG.info("Splitting data")
    train, valid, test = data_formatter.split_data(df, d1850, d1913)

    LOG.info("getting experiment params")
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
        output_folder, 'saved_models', 'walmart', 'fixed')
    model_params['model_folder'] = model_folder

    model_params.update(data_params)
    """# Specify GPU usage
    if USE_GPU:
        tf_config = utils.get_default_tensorflow_config(
            tf_device="gpu", gpu_id=0)
    else:
        tf_config = utils.get_default_tensorflow_config(
            tf_device="cpu", gpu_id=0)"""
    LOG.info("Starting Training")

    model = run_train()
    LOG.info("Training complete")
    tf.compat.v1.reset_default_graph()

    LOG.info("starting evaluation")
    with tf.Graph().as_default(), tf.Session(config=tf_config) as sess:

        tf.keras.backend.set_session(sess)

        # Create a new model & load weights
        model = TemporalFusionTransformer(model_params,
                                          use_cudnn=True)
        LOG.info("loading model from {model_folder}")
        model.load(model_folder)

        # Make forecasts
        LOG.info("Starting predictions")

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
    LOG.info("Normalised quantile losses: P50={}, P90={}".format(
        p50_loss.mean(), p90_loss.mean()))
    LOG.info("--- walmart-sales.py done --")
