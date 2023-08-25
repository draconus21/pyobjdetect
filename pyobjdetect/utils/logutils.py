import os
import json
import click
import logging
import logging.config
import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy


DEBUG = 0
INFO = 1
WARN = 2
ERROR = 3

VERBOSE = INFO


def makeDictJsonReady(dictData: dict):
    def sanitize(value):
        if type(value) in [np.float, np.float16, np.float32, np.float64]:
            return float(value)
        elif isinstance(value, list):
            t = [0] * len(value)
            for i, v in enumerate(value):
                t[i] = sanitize(v)
            return t
        elif isinstance(value, np.ndarray):
            return value.tolist()
        else:
            return value

    jsonDict = deepcopy(dictData)
    for key, value in dictData.items():
        if isinstance(value, dict):
            jsonDict[key] = makeDictJsonReady(value)
        else:
            jsonDict[key] = sanitize(value)
    return jsonDict


def prettyDumpDict(dictData):
    return json.dumps(makeDictJsonReady(dictData), indent=4, sort_keys=True)


def error(message, verboseLvl=3):
    secho(message, fg="red", verboseLvl=verboseLvl)
    logging.error(message)


def warn(message, verboseLvl=2):
    secho(message, fg="yellow", verboseLvl=verboseLvl)
    logging.warning(message)


def info(message, verboseLvl=1):
    secho(message, verboseLvl=verboseLvl)
    logging.info(message)


def debug(message, verboseLvl=0):
    secho(message, fg=None, verboseLvl=verboseLvl)
    logging.debug(message)


def secho(message, fg="cyan", verboseLvl=0):
    if verboseLvl < VERBOSE:
        return  # do nothing

    if isinstance(message, str):
        click.secho(message, fg=fg)
    elif isinstance(message, dict):
        click.secho(json.dumps(message, indent=" " * 4), fg=fg)


def setupLogging(level: str = "INFO", env_key: str = "ODT_LOG_CFG"):
    """
    Setup logging
    """
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    plt.style.use("seaborn-pastel")

    level = level.upper()
    value = os.getenv(env_key, None)
    logstr = []
    path = None
    if value is not None:
        path = value
    if path is not None and os.path.exists(path):
        with open(path, "rt") as f:
            config = json.load(f)
            config["root"]["level"] = level
        logging.config.dictConfig(config)
        logstr.append(f"logging init from provided config: {path}")
    else:
        logging.basicConfig(level=level)
        logstr.append(f"using basicConfig")

    logging.info("\n".join(logstr))
