import json

from src.common_functions.run_lambda_grid_search import grid_search_lambda, do_grid_seach_model
from src.inflow_forecast.model.point_and_sarx.model_point import ModelPointBase
from src.common_functions.get_data import get_data_lags_thresholds_w_interaction
from src.common_functions.misc import params_to_json, get_path_make_dir
from os.path import join
from src.config_variables import *
from src.config_paths import *

SEASONS_DAYS_WEEKS = [[0, 1, 2, 3, 4], [5], [6]]
N_SEASONS_ANNUAL = 4




# lbda_start, class model, min_lag, thresh_pred, thresh_ext, thresh_rain, thresh_interactions, use_future_rain, use_rain_forecasts, steps_cum_rain, interactions_only_known_data, lags_all_future_rain, interaction rain with rain
cls = ModelPointBase
data_models = [
    ### Oracles
    ( 1e-5,     cls,    min_lag,    1,      0,      0,      0,      False,      False,      None,       False,      False,      False),
    ( 1e-5,     cls,    min_lag,    1,      1,      0,      0,      False,      False,      None,       False,      False,      False),
    ( 1e-5,     cls,    min_lag,    1,      1,      1,      0,      False,      False,      None,       False,      False,      False),
    ( 1e-5,     cls,    min_lag,    1,      1,      1,      0,      True,       False,      None,       False,      False,      False),
    ( 1e-5,     cls,    min_lag,    3,      3,      3,      0,      True,       False,      None,       False,      False,      False),
    ( 1e-5,     cls,    min_lag,    15,     15,     5,      0,      True,       False,      None,       False,      False,      False),
    ( 1e-5,     cls,    min_lag,    15,     15,     5,      0,      True,       False,      6,          False,      False,      False),
    ( 1e-5,     cls,    min_lag,    15,     15,     5,      3,      True,       False,      None,       False,      False,      False),
    ( 1e-5,     cls,    min_lag,    15,     15,     5,      3,      True,       False,      6,          False,      False,      False),
    ### Rain Forecast
    ( 1e-5,     cls,    min_lag,    15,     15,     5,      3,      True,       True,       6,          False,      True,       True),
]

lags_target_additional = []

cols_nm = [name_column_rain_history]

dir_save_models = get_path_make_dir(dir_models, cls.__name__)
dir_save_results = get_path_make_dir(dir_grid_search, cls.__name__)
for args in data_models:
    if len(args) == 11:
        # Add furain all lags and rain rain inter
        args = (*args, False, False)
    do_grid_seach_model(
        date_start_train,
        date_end_train,
        date_start_test,
        date_end_test,
        lags_target_additional,
        name_column_inflow,
        variables_external,
        cols_nm,
        dir_save_results,
        dir_save_models,
        *args,
        N_SEASONS_ANNUAL,
        SEASONS_DAYS_WEEKS
    )