import os
import uuid

import pandas as pd


class BaseLogger:
    """Base logger class providing common path and name handling.
    """
    def __init__(self, log_path, log_file_name='history'):

        self.log_path = os.path.abspath(log_path)
        self.log_file_name = log_file_name


class CsvLogger(BaseLogger):
    """Uses pandas Dataframe to process history
    and store it as a csv.
    """

    def save_history(self, data, run_id):
        """Saves run history as csv file with name consisting of prefix
        set during the __init__ call and a id hash of the data object.
        Raises PermissionError if denied access.
        """
      

        log_file_name = "{0}_{1}.log".format(
            self.log_file_name,
            run_id)

        path_to_file = os.path.join(self.log_path, log_file_name)

        log = pd.DataFrame(
            data=data
        )

        try:
            log.to_csv(path_or_buf=path_to_file)
        except PermissionError:
            raise PermissionError()
