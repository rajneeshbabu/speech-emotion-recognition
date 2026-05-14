"""
Custom exception with traceback detail.
"""
import sys
import traceback


def _error_message(error: Exception) -> str:
    _, _, tb = sys.exc_info()
    if tb is None:
        return str(error)
    fname = tb.tb_frame.f_code.co_filename
    lineno = tb.tb_lineno
    return (
        f"Error in [{fname}] at line [{lineno}]:\n"
        f"  {type(error).__name__}: {error}\n"
        f"{''.join(traceback.format_tb(tb))}"
    )


class SERException(Exception):
    def __init__(self, error: Exception):
        super().__init__(_error_message(error))
