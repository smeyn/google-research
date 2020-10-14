# coding=utf-8
# Copyright 2020 Google.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Custom formatting functions for Volatility dataset.

Defines dataset specific column definitions and data transformations.
"""

import data_formatters.base
import libs.utils as utils
import sklearn.preprocessing
import logging

LOG = logging.getLogger(__name__)

GenericDataFormatter = data_formatters.base.GenericDataFormatter
DataTypes = data_formatters.base.DataTypes
InputTypes = data_formatters.base.InputTypes


class SalesFormatter(GenericDataFormatter):
    """Defines and formats data for the volatility dataset.

    Attributes:
      column_definition: Defines input and data type of column used in the
        experiment.
      identifiers: Entity identifiers used in experiments.

      Available data types:
          DataTypes.REAL_VALUED
          DataTypes.CATEGORICAL
          DataTypes.DATE

      Available input types:
          InputTypes.TARGET
          InputTypes.OBSERVED_INPUT
          InputTypes.KNOWN_INPUT
          InputTypes.STATIC_INPUT
          InputTypes.ID
          InputTypes.TIME
    """

    _column_definition = [
        ('date', DataTypes.DATE, InputTypes.TIME),
        # ('id', DataTypes., InputTypes.),
        ('value', DataTypes.REAL_VALUED, InputTypes.TARGET),
        # ('wm_yr_wk', DataTypes., InputTypes.),
        # ('weekday', DataTypes., InputTypes.),
        # ('wday', DataTypes., InputTypes.),
        # ('month', DataTypes., InputTypes.),
        # ('year', DataTypes., InputTypes.),
        ('event_name_1', DataTypes.CATEGORICAL, InputTypes.OBSERVED_INPUT),
        ('event_type_1', DataTypes.CATEGORICAL, InputTypes.OBSERVED_INPUT),
        ('event_name_2', DataTypes.CATEGORICAL, InputTypes.OBSERVED_INPUT),
        ('event_type_2', DataTypes.CATEGORICAL, InputTypes.OBSERVED_INPUT),
        ('snap_CA', DataTypes.CATEGORICAL, InputTypes.OBSERVED_INPUT),
        ('snap_TX', DataTypes.CATEGORICAL, InputTypes.OBSERVED_INPUT),
        ('snap_WI', DataTypes.CATEGORICAL, InputTypes.OBSERVED_INPUT),
        ('store_id', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
        ('item_id', DataTypes.CATEGORICAL, InputTypes.ID),
        ('sell_price', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        ('cat_id', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
        ('dept_id', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
        ('state_id', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
        # ('week', DataTypes., InputTypes.),
        # ('day', DataTypes., InputTypes.),
        # ('quarter', DataTypes., InputTypes.)
    ]

    def __init__(self):
        """Initialises formatter."""

        self.identifiers = None
        self._real_scalers = None
        self._cat_scalers = None
        self._target_scaler = None
        self._num_classes_per_cat_input = None

    def split_data(self, df, valid_boundary, test_boundary):
        """Splits data frame into training-validation-test data frames.

        This also calibrates scaling object, and transforms data for each split.

        Args:
          df: Source data frame to split.
          valid_boundary: Starting year for validation data
          test_boundary: Starting year for test data

        Returns:
          Tuple of transformed (train, valid, test) data.
        """
        LOG.info('Formatting train-valid-test splits.')

        index = df['date']
        train = df.loc[index < valid_boundary]
        valid = df.loc[(index >= valid_boundary) & (index < test_boundary)]
        test = df.loc[index >= test_boundary]

        self.set_scalers(train)

        return (self.transform_inputs(data) for data in [train, valid, test])

    def set_scalers(self, df):
        """Calibrates scalers using the data supplied.

        Args:
          df: Data to use to calibrate scalers.
        """
        LOG.info('Setting scalers with training data...')

        column_definitions = self.get_column_definition()
        id_column = utils.get_single_col_by_input_type(InputTypes.ID,
                                                       column_definitions)
        target_column = utils.get_single_col_by_input_type(InputTypes.TARGET,
                                                           column_definitions)

        # Extract identifiers in case required
        self.identifiers = list(df[id_column].unique())
        LOG.info("format real scalers")
        # Format real scalers
        real_inputs = utils.extract_cols_from_data_type(
            DataTypes.REAL_VALUED, column_definitions,
            {InputTypes.ID, InputTypes.TIME})

        data = df[real_inputs].values
        self._real_scalers = sklearn.preprocessing.StandardScaler().fit(data)
        self._target_scaler = sklearn.preprocessing.StandardScaler().fit(
            df[[target_column]].values)  # used for predictions

        # Format categorical scalers
        LOG.info("format categorical scalers")
        categorical_inputs = utils.extract_cols_from_data_type(
            DataTypes.CATEGORICAL, column_definitions,
            {InputTypes.ID, InputTypes.TIME})

        categorical_scalers = {}
        num_classes = []
        for col in categorical_inputs:
            LOG.info(f"format categorical scaler {col}")
            # Set all to str so that we don't have mixed integer/string columns
            srs = df[col].apply(str)
            categorical_scalers[col] = sklearn.preprocessing.LabelEncoder().fit(
                srs.values)
            num_classes.append(srs.nunique())

        # Set categorical scaler outputs
        LOG.info("Set categorical scaler outputs")
        self._cat_scalers = categorical_scalers
        self._num_classes_per_cat_input = num_classes
        LOG.info("set_scalers done")

    def transform_inputs(self, df):
        """Performs feature transformations.

        This includes both feature engineering, preprocessing and normalisation.

        Args:
          df: Data frame to transform.

        Returns:
          Transformed data frame.

        """
        LOG.info("transform_inputs")
        output = df.copy()

        if self._real_scalers is None and self._cat_scalers is None:
            raise ValueError('Scalers have not been set!')

        column_definitions = self.get_column_definition()

        real_inputs = utils.extract_cols_from_data_type(
            DataTypes.REAL_VALUED, column_definitions,
            {InputTypes.ID, InputTypes.TIME})
        categorical_inputs = utils.extract_cols_from_data_type(
            DataTypes.CATEGORICAL, column_definitions,
            {InputTypes.ID, InputTypes.TIME})

        # Format real inputs
        LOG.info("Format real inputs")
        output[real_inputs] = self._real_scalers.transform(
            df[real_inputs].values)

        LOG.info("Format categorical inputs")
        # Format categorical inputs
        for col in categorical_inputs:
            LOG.info(f"Transforming {col} - converting to str")
            string_df = df[col].apply(str)
            LOG.info(f"Transforming {col} - set scaler")
            output[col] = self._cat_scalers[col].transform(string_df)
        LOG.info("transform inputs - DONE")
        return output

    def format_predictions(self, predictions):
        """Reverts any normalisation to give predictions in original scale.

        Args:
          predictions: Dataframe of model predictions.

        Returns:
          Data frame of unnormalised predictions.
        """
        LOG.info("format_predictions")
        output = predictions.copy()

        column_names = predictions.columns

        for col in column_names:
            if col not in {'forecast_time', 'identifier'}:
                output[col] = self._target_scaler.inverse_transform(
                    predictions[col])

        LOG.info("format Predictions - DONE")
        return output

    # Default params
    def get_fixed_params(self):
        """Returns fixed model parameters for experiments."""

        fixed_params = {
            'total_time_steps': 252 + 5,
            'num_encoder_steps': 252,
            'num_epochs': 100,
            'early_stopping_patience': 5,
            'multiprocessing_workers': 5,
        }

        return fixed_params

    def get_default_model_params(self):
        """Returns default optimised model parameters."""

        model_params = {
            'dropout_rate': 0.3,
            'hidden_layer_size': 160,
            'learning_rate': 0.01,
            'minibatch_size': 64,
            'max_gradient_norm': 0.01,
            'num_heads': 1,
            'stack_size': 1
        }

        return model_params
