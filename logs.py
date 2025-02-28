import logging
import logging.config


def init_logging():
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "json": {
                    "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
                    "format": "%(asctime)s %(levelname)s %(filename)s %(lineno)s %(message)s",
                },
                "color": {
                    "format": "%(name)16s\t %(message)s",
                },
            },
            "handlers": {
                "json": {
                    "level": "DEBUG",
                    "class": "logging.StreamHandler",
                    "formatter": "json",
                },
                "color": {
                    "level": "DEBUG",
                    "class": "rich.logging.RichHandler",
                    "formatter": "color",
                },
            },
            "loggers": {
                "__main__": {  # if you want to log from your script
                    "handlers": ["json"],
                    "level": "INFO",
                    "propagate": True,
                },
                "chat": {  # if you want to log from your script
                    "handlers": ["color"],
                    "level": "INFO",
                    "propagate": True,
                },
                "thoughts": {  # if you want to log from your script
                    "handlers": ["color"],
                    "level": "INFO",
                    "propagate": True,
                },
                "actions": {  # if you want to log from your script
                    "handlers": ["color"],
                    "level": "INFO",
                    "propagate": True,
                },
                "textworld": {  # if you want to log from your script
                    "handlers": ["json"],
                    "level": "INFO",
                    "propagate": True,
                },
                "relative_world_ollama": {  # if you want to log from your script
                    "handlers": ["json"],
                    "level": "WARN",
                    "propagate": True,
                },
                "relative_world": {  # if you want to log from your script
                    "handlers": ["json"],
                    "level": "WARN",
                    "propagate": True,
                },
            },
        }
    )
