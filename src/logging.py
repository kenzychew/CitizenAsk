"""Logging setup for CitizenAsk: a YAML logging config with a fallback."""

import logging
import logging.config
import os

import yaml

logger = logging.getLogger(__name__)


def setup_logging(
    logging_config_path: str = "configs/logging.yaml",
    default_level: int = logging.INFO,
    log_dir: str | None = None,
) -> None:
    """Configure logging from a YAML file, falling back to basicConfig.

    Args:
        logging_config_path: Path to the YAML logging config.
        default_level: Level used by the fallback basicConfig.
        log_dir: If set, redirect every file handler's filename into this
            directory instead of the one named in the config.
    """
    try:
        with open(logging_config_path, encoding="utf-8") as file:
            log_config = yaml.safe_load(file.read())

        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            for handler in log_config.get("handlers", {}).values():
                if "filename" in handler:
                    filename = os.path.basename(handler["filename"])
                    handler["filename"] = os.path.join(log_dir, filename)

        logging.config.dictConfig(log_config)

    except (FileNotFoundError, yaml.YAMLError, ValueError) as error:
        logging.basicConfig(
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            level=default_level,
        )
        logger.error("Logging config file could not be loaded, using basic config: %s", error)
