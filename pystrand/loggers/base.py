import os

class BaseLogger:
    """Base logger class providing common path and name handling.
    """
    def __init__(self, log_path, log_file_name='history'):

        self.log_path = os.path.abspath(log_path)
        self.log_file_name = log_file_name
