import sys
import traceback

class CustomException(Exception):
    def __init__(self, message, error_detail: sys):
        super().__init__(message)

        # Try to extract traceback info safely
        _, _, tb = sys.exc_info()

        if tb is not None:
            self.filename = tb.tb_frame.f_code.co_filename
            self.line = tb.tb_lineno
        else:
            self.filename = "Unknown File"
            self.line = "Unknown Line"

        self.message = message

    def __str__(self):
        return f"Error in script [{self.filename}] at line [{self.line}]: {self.message}"
