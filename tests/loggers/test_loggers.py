from unittest import TestCase, mock

import pystrand.loggers as loggers

class TestCsvLogger(TestCase):

    def setUp(self):
        super(TestCsvLogger, self).setUp()

    @mock.patch(
        'pystrand.loggers.csv_logger.os.path.abspath',
        return_value='foo/foo/bar')
    def test_logger_init(self, mock_abspath):

        logger = loggers.csv_logger.CsvLogger(
            'foo/bar',
            'fizz')

        self.assertEqual('foo/foo/bar', logger.log_path)
        self.assertEqual('fizz', logger.log_file_name)

    @mock.patch(
        'pystrand.loggers.csv_logger.pd.DataFrame')
    def test_save_history_success(self, mock_dataframe):

        logger = loggers.csv_logger.CsvLogger(
            'foo/bar',
            'fizz')

        data = {'foo': 'bar'}

        logger.save_history(data, 'buzz')

        mock_dataframe.assert_called_once_with(data=data)
