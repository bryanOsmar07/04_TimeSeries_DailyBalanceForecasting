import sys

from src.logger import logger


def error_message_detail(error, error_detail: sys):
    """
    Genera un mensaje de error detallado con:
    - nombre del script
    - número de línea
    - mensaje original del error
    """
    _, _, exc_tb = error_detail.exc_info()

    file_name = exc_tb.tb_frame.f_code.co_filename if exc_tb else "Unknown file"
    line_no = exc_tb.tb_lineno if exc_tb else -1

    error_message = (
        f"Error occurred in python script [{file_name}] at line [{line_no}] "
        f"- message: [{error}]"
    )

    return error_message


class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        """
        Envuelve una excepción original con más contexto y lo deja logueado.
        """
        super().__init__(error_message)
        self.error_message = error_message_detail(
            error_message, error_detail=error_detail
        )

        # Logueamos el error apenas se crea la excepción
        logger.error(self.error_message)

    def __str__(self):
        return self.error_message
