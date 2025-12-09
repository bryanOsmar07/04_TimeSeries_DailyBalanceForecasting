# tests/test_utils.py

import pytest
from src.utils import save_object, load_object
from src.exception import CustomException


def test_save_and_load_object(tmp_path):
    """
    Verifica que save_object y load_object funcionen correctamente:
    - Crea el directorio si no existe
    - Guarda el objeto
    - Lo carga y es igual al original
    """
    obj = {"a": 1, "b": [1, 2, 3], "c": "test"}

    file_path = tmp_path / "subdir" / "test_obj.pkl"

    # Guardar
    save_object(str(file_path), obj)

    # Aseguramos que el archivo existe
    assert file_path.exists()

    # Cargar
    loaded = load_object(str(file_path))

    assert loaded == obj


def test_load_object_nonexistent_raises_customexception(tmp_path):
    """
    Verifica que cargar un archivo inexistente dispare CustomException.
    """
    fake_path = tmp_path / "no_existe.pkl"

    with pytest.raises(CustomException):
        load_object(str(fake_path))
