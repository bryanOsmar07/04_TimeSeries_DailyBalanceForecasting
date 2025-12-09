from typing import List

from setuptools import find_packages, setup

HYPEN_E_DOT = "-e ."


def get_requirements(file_path: str) -> List[str]:
    """
    this function will return the list of requirements
    """
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements


setup(
    name="TimeSeries_DailyBalanceForecasting",
    version="0.0.1",
    author="bryanOsmar07",
    author_email="bryanosmar07@gmail.com",
    description="Forecasting de variaciones diarias de saldos mediante ML.",
    url="https://github.com/bryanOsmar07/04_TimeSeries_DailyBalanceForecasting",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)
