import logging


class AppLogger:
    """
    Application-wide logging configuration.
    """

    def __init__(
        self,
        name: str = "document_analyzer",
    ):
        self.logger = logging.getLogger(name)

        if not self.logger.handlers:
            handler = logging.StreamHandler()

            formatter = logging.Formatter(
                "%(asctime)s | "
                "%(levelname)s | "
                "%(name)s | "
                "%(message)s"
            )

            handler.setFormatter(formatter)

            self.logger.addHandler(handler)

            self.logger.setLevel(
                logging.INFO
            )

    def info(self, message: str):
        self.logger.info(message)

    def warning(self, message: str):
        self.logger.warning(message)

    def error(self, message: str):
        self.logger.error(message)

    def debug(self, message: str):
        self.logger.debug(message)


app_logger = AppLogger()