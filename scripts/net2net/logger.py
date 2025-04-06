import sys


class Logger:
    """
    Logger class that duplicates console output to a log file.

    This class redirects all standard output (stdout) to both the
    terminal and a specified log file, allowing real-time console
    monitoring and persistent logging.

    Args:
        log_path (str): Path to the file where logs will be saved.
    """

    def __init__(self, log_path):
        """
        Initialize the Logger.

        Args:
            log_path (str): The path to the log file where outputs
            will be written.
        """
        self.terminal = sys.stdout
        self.log = open(log_path, "a")

    def write(self, message):
        """
        Write a message to both the terminal and the log file.

        Args:
            message (str): The message to write.
        """
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        """
        Flush the output buffers of the terminal and the log file.
        """
        self.terminal.flush()
        self.log.flush()
