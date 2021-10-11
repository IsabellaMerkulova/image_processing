import time
import logging

logger = logging.getLogger()
logger.setLevel('INFO')


# decorator
def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        return result, end - start
    return wrapper
