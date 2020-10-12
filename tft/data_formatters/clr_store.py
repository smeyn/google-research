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
import data_formatters.base


class StoreFormatter(base.GenericDataFormatter):
    """Defines and formats data for the Store dataset.

  data is based on this bq view

  `gcp-wow-pel-ml-mpe-dev.matapae.v_receipt_line_sales_to_department`
  `gcp-wow-pel-ml-mpe-dev.matapae.v_receipt_line_sales_to_artcile`

    This also performs z-score normalization across the entire dataset, hence
    re-uses most of the same functions as volatility.

    Attributes:
      column_definition: Defines input and data type of column used in the
        experiment.
      identifiers: Entity identifiers used in experiments.
    """

    _column_definition = [
        ('department_id', DataTypes.CATEGORICAL, InputTypes.ID),
        #('department_name', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
        ('receipt_date', DataTypes.REAL_VALUED, InputTypes.TIME),
        ('district_id', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
        #('district_name', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
        ('category_id_int'	, DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
        ('sub_category_id_int'	, DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
        ('segment_id_int'	, DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
        ('article_quantity', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        ('article_gst_excluded_amount', DataTypes.REAL_VALUED, InputTypes.TARGET),
    ]

    def split_data(self, df, valid_boundary='2019-01-31', test_boundary='2019-03-04',
                   end_boundary='2019-03-10'):
        """Splits data frame into training-validation-test data frames.

        This also calibrates scaling object, and transforms data for each split.

        Args:
          df: Source data frame to split.
          valid_boundary: Starting date for validation data
          test_boundary: Starting date for test data
          end_boudnary: end date for all data

        Returns:
          Tuple of transformed (train, valid, test) data.
        """

        print('Formatting train-valid-test splits.')

        index = df['receipt_date']
        train = df.loc[index < valid_boundary]
        valid = df.loc[(index >= valid_boundary - 7)
                       & (index < test_boundary)]
        test = df.loc[index >= test_boundary - 7 & (index < end_boundary)]

        self.set_scalers(train)

        return (self.transform_inputs(data) for data in [train, valid, test])

    # Default params
    def get_fixed_params(self):
        """Returns fixed model parameters for experiments."""

        fixed_params = {
            'total_time_steps': 8 * 24,
            'num_encoder_steps': 7 * 24,
            'num_epochs': 100,
            'early_stopping_patience': 5,
            'multiprocessing_workers': 5
        }

        return fixed_params

    def get_default_model_params(self):
        """Returns default optimised model parameters."""

        model_params = {
            'dropout_rate': 0.3,
            'hidden_layer_size': 320,
            'learning_rate': 0.001,
            'minibatch_size': 128,
            'max_gradient_norm': 100.,
            'num_heads': 4,
            'stack_size': 1
        }

        return model_params

    def get_num_samples_for_calibration(self):
        """Gets the default number of training and validation samples.

        Use to sub-sample the data for network calibration and a value of -1 uses
        all available samples.

        Returns:
          Tuple of (training samples, validation samples)
        """
        return 4500, 500
