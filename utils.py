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
        tc = end - start
        logging.info(f'Time consumed {tc}')
        return result, tc
    return wrapper
