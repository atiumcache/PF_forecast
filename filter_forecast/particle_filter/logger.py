import logging
from datetime import datetime

import pytz

from filter_forecast.particle_filter.init_settings import InitSettings


def get_logger():
    timezone = pytz.timezone("US/Arizona")
    time_now = datetime.now(tz=timezone)
    logger = logging.get_Logger(__name__)
    logging.basicConfig(
        filename=f"./output/{time_now}output.log", encoding="utf-8", level=logging.DEBUG
    )
    return logger
