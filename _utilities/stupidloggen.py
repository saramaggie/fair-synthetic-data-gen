import logging
import os
import sys
from datetime import datetime

DIR_PATH = os.environ["VIRTUAL_ENV"][:-5]
sys.path.append(DIR_PATH)


class StupidLogGenerator:
    """
    Simple logger generator that creates a logger which logs to both a file and the console.
    The log file is stored in a specified folder with a timestamped name.

    Returns:
        Logger: initialized logger instance
    """

    @staticmethod
    def initLogger(log_folder="_logs"):
        log_folder = f"{DIR_PATH}/{log_folder}"
        root_logger = logging.getLogger(__name__)
        root_logger.setLevel(logging.INFO)

        if not os.path.exists(log_folder):
            os.makedirs(log_folder)

        # File name
        file_name = f"log-{datetime.now().strftime('%Y-%m-%d')}"

        # Strings for log formatting
        msg_format_file = "%(asctime)s [%(funcName)s] [%(levelname)s]  %(message)s"
        msg_format_console = "[%(levelname)s]: %(message)s"

        # Create a file handler to write logs to a file
        file_handler = logging.FileHandler(f"{log_folder}/{file_name}.log")
        file_handler.setFormatter(logging.Formatter(msg_format_file))
        root_logger.addHandler(file_handler)

        # Create a stream handler to print logs to the console
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(msg_format_console))
        root_logger.addHandler(console_handler)

        return root_logger
