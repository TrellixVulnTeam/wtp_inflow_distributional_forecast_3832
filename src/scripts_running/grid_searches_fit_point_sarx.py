import json
from src.common_functions.run_lambda_grid_search import do_grid_seach_model
from src.inflow_forecast.model.point_and_sarx.model_point import ModelPointBase
from src.common_functions.get_data import get_data_lags_thresholds_w_interaction
from src.common_functions.misc import params_to_json
from os.path import join
from src.config_variables import *
from src.config_paths import *

SEASONS_DAYS_WEEKS = [[0, 1, 2, 3, 4, 5, 6]]
N_SEASONS_ANNUAL = 1

# lbda_start, class model, min_lag, thresh_pred, thresh_ext, thresh_rain, thresh_interactions, use_future_rain, use_rain_forecasts, steps_cum_rain, interactions_only_known_data
cls = ModelPointBase
min_lag = -6
data_models = [
    (
        1e-4,
        cls,
        -1,
        1, 1, 1, 0,
        True,
        False,
        None,
        False,
        False,
        False
    ),
    (
        1e-4,
        cls,
        -1,
        1, 1, 1, 0,
        True,
        True,
        None,
        False,
        False,
        False
    ),
]


lags_target_additional = range(-6, 0)

dir_save_models = os.path.join(dir_models, cls.__name__)
dir_save_results = os.path.join(dir_grid_search, cls.__name__)
for args in data_models:
    do_grid_seach_model(
        date_start_train,
        date_end_train,
        date_start_test,
        date_end_test,
        lags_target_additional,
        name_column_inflow,
        variables_external,
        [name_column_rain_history],
        dir_save_results,
        dir_save_models,
        *args,
        N_SEASONS_ANNUAL,
        SEASONS_DAYS_WEEKS,
        'sarx_'
    )