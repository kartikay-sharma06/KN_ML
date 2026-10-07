import sys
import logging


def error_msg_detail(error, error_detail):
    _, _, exc_tb = error_detail

    file_name = exc_tb.tb_frame.f_code.co_filename

    error_message = (
        f"Error occurred in Python script name [{file_name}] "
        f"line number [{exc_tb.tb_lineno}] "
        f"error message [{error}]"
    )

    return error_message


class CustomException(Exception):

    def __init__(self, error_message, error_detail):
        super().__init__(error_message)

        self.error_message = error_msg_detail(
            error_message,
            error_detail
        )

    def __str__(self):
        return self.error_message

