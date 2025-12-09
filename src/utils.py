# src/utils.py

import os
import pickle
import sys
from typing import Any

from src.exception import CustomException
from src.logger import logger


def save_object(file_path: str, obj: Any) -> None:
    """
    Guarda cualquier objeto serializable en el path indicado usando pickle.
    - Crea los directorios si no existen
    - Maneja excepciones propias del pipeline
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

        logger.info(f"Objeto guardado en: {file_path}")

    except Exception as e:
        logger.error(f"Error al guardar objeto en {file_path}")
        raise CustomException(e, sys)


def load_object(file_path: str) -> Any:
    """
    Carga un objeto pickle desde el path especificado.
    """
    try:
        with open(file_path, "rb") as file_obj:
            logger.info(f"Objeto cargado desde: {file_path}")
            return pickle.load(file_obj)

    except Exception as e:
        logger.error(f"Error al cargar objeto desde {file_path}")
        raise CustomException(e, sys)
