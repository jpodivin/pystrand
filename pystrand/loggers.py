import os
import pandas as pd

class CsvLogger:
    """Uses pandas Dataframe to process history
    and store it as a csv.
    """

    def __init__(self, log_path, log_file_name='history'):
        """
        """
        self.log_path = os.path.abspath(log_path)
        self.log_file_name = log_file_name

    def save_history(self, data):
        """
        """
        log = pd.DataFrame(
            data=data
        )

        log_file_name = self.log_file_name + '_' + str(id(data)) + '.log'

        path_to_file = os.path.join(self.log_path, log_file_name)

        log.to_csv(path_or_buf=path_to_file)