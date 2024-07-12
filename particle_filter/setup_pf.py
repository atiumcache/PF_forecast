from particle_filter.init_settings import InitSettings
import logging
from datetime import datetime
import pytz


def get_settings(state_population: int, loc_code: str) -> InitSettings:
    settings = InitSettings(
        num_particles=1000, population=state_population, location_code=loc_code
    )
    return settings


def get_logger():
    timezone = pytz.timezone("US/Arizona")
    time_now = datetime.now(tz=timezone)
    logger = logging.get_Logger(__name__)
    logging.basicConfig(
        filename=f"./output/{time_now}output.log", encoding="utf-8", level=logging.DEBUG
    )
    return logger
