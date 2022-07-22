import functools

import wandb
from absl import app, flags
from eigengame import models
from jax.profiler import trace
from jaxline import platform

_BATCH_SIZE = flags.DEFINE_integer("batch_size", 0, "batch size")
_MODEL = flags.DEFINE_string("model", "game", "model")
_DATA = flags.DEFINE_string("data", "linear", "dataset name")
_N_COMPONENTS = flags.DEFINE_integer("n_components", 4, "number of components")
_LEARNING_RATE = flags.DEFINE_float("learning_rate", 1e-3, "learning rate")
_EPOCHS = flags.DEFINE_integer("epochs", 100, "epochs")
_LOGGING_INTERVAL = flags.DEFINE_float("logging_interval", 1, "logging interval")

MODEL_DICT = {
        "game": models.Game,
        "gha": models.GHA,
        "sgha": models.SGHA,
        "oja": models.Oja,
    }

defaults = {
    "model": _MODEL.default,
    "data": _DATA.default,
    "epochs": _EPOCHS.default,
    "batch_size": _BATCH_SIZE.default,
    "n_components": _N_COMPONENTS.default,
    "learning_rate": _LEARNING_RATE.default,
    "logging_interval": _LOGGING_INTERVAL.default,
}

if __name__ == "__main__":
    wandb.init(config=defaults, sync_tensorboard=True)
    Experiment = MODEL_DICT[wandb.config.model]
    #with trace("/tmp/jax-trace", create_perfetto_link=True):
    app.run(functools.partial(platform.main, Experiment))
