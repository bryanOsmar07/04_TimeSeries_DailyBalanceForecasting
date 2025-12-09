import os
import pandas as pd
import pytest
from datetime import datetime, timedelta


@pytest.fixture
def sample_raw_df():
    """Un mini dataset de prueba para ingestion & transformation."""
    base_date = datetime(2024, 1, 1)
    data = {
        "load_date": [base_date + timedelta(days=i) for i in range(10)],
        "saldo": [1000 + i * 10 for i in range(10)],
    }
    return pd.DataFrame(data)


@pytest.fixture
def tmp_artifacts_dir(tmp_path):
    """Carpeta temporal para simular artifacts sin tocar los reales."""
    d = tmp_path / "artifacts"
    d.mkdir()
    return d
