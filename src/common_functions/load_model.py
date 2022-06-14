from src.inflow_forecast.model.model_base import ModelBase
from src.inflow_forecast.model.benchmark.model_rnn_separate_steps import ModelRnnSeparateTimestepsJsu, \
    ModelRnnSeparateTimesteps
from src.inflow_forecast.model.benchmark.model_benchmark_base import ModelBenchmarkBase
from src.inflow_forecast.model.point_and_sarx.model_point import ModelPointBase
from src.inflow_forecast.model.gamlss.models_gamlss import ModelGamlssJsu, ModelGamlssGaussian
from src.inflow_forecast.model.gamlss.model_gamlss_base import ModelGamlssBase
import json
from src.constants.misc import KWARGS_MODEL


def get_model_class_file(path_model):
    classes = {
        cls.__name__: cls
        for cls in [
            ModelBase,
            ModelGamlssGaussian,
            ModelRnnSeparateTimesteps,
            ModelGamlssBase,
            ModelGamlssJsu,
            ModelPointBase,
            ModelBenchmarkBase,
            ModelRnnSeparateTimestepsJsu
        ]
    }

    with open(path_model) as f:
        model_json = json.load(f)

    name_class = model_json[KWARGS_MODEL]['class_model']

    return classes[name_class]


def load_model_file_get_class(path_model, *args, **kwargs):
    class_model = get_model_class_file(path_model)
    model = class_model.model_from_file(
        path_model,
        *args,
        *kwargs
    )

    return model
