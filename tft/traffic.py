import pandas as pd
import os
from script_download_data import main as download_data
from libs.tft_model import TemporalFusionTransformer
from data_formatters.base import GenericDataFormatter, DataTypes, InputTypes
from libs import utils        # Load TFT helper functions
import sklearn.preprocessing  # Used for data standardization
from TrafficFormatter import TrafficFormatter
import os
import tensorflow as tf
import logging

logging.basicConfig(level=logging.DEBUG, filenam="debuglog.txt")
USE_GPU = False


def download_parameters():
    # Download parameters
    expt_name = "traffic"
    # Name of default experiment
    # Root folder to save experiment outputs
    output_folder = os.path.join(os.getcwd(), 'outputs')
    force_download = False  # Skips download if data is already present

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Downloads main csv file if not present
    csv_file = os.path.join(output_folder, 'data',
                            'traffic', 'hourly_data.csv')
    if not os.path.exists(csv_file):
        download_data(expt_name, force_download=True,
                      output_folder=output_folder)
    converters = {
       0: int, 1:float,2: float,3: float,4: int, 5:int, 6:int,7: int, 8:int, 9:float,10: int,11:int
    }
    # Load the downloaded data
    df = pd.read_csv(csv_file, index_col=0, converters = converters, na_filter=False)
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
        model.save(model_folder)
    return model


if __name__ == '__main__':
    logging.info("Starting")
    df, output_folder = download_parameters()
    # Create a data formatter
    data_formatter = TrafficFormatter()

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

    tf.compat.v1.reset_default_graph()
    with tf.Graph().as_default(), tf.Session(config=tf_config) as sess:

        tf.keras.backend.set_session(sess)

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
