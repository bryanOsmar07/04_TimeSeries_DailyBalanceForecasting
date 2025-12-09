# tests/test_data_transformation.py

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation


def test_data_transformation():
    """
    Prueba que DataTransformation genere features y splits
    a partir del raw de data_ingestion.
    """
    # Aseguramos que exista raw.csv y splits básicos
    ingestion = DataIngestion()
    ingestion.initiate_data_ingestion()

    transformer = DataTransformation()
    result = transformer.initiate_data_transformation()

    # Compatibilidad: según tu implementación puede devolver 4 o 7 elementos
    if len(result) == 4:
        X_train, X_val, X_test, feature_cols = result
    elif len(result) == 7:
        X_train, X_val, X_test, y_train, y_val, y_test, feature_cols = result
    else:
        raise AssertionError(
            f"initiate_data_transformation devolvió {len(result)} elementos (esperaba 4 o 7)."
        )

    # Checks suaves: que haya filas y haya columnas definidas
    assert hasattr(X_train, "shape")
    assert X_train.shape[0] > 0
    assert hasattr(X_val, "shape")
    assert hasattr(X_test, "shape")

    assert isinstance(feature_cols, (list, tuple))
    assert len(feature_cols) > 0
