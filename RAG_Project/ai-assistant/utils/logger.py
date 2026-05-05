import logging

def get_logger(name):
    logger = logging.getLogger(name)
    logging.basicConfig(level=logging.INFO)
    return logger
