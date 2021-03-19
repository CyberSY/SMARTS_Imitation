from setuptools import setup

setup(
    name="smarts_imitation",
    version="0.1",
    install_requires=["gym", "smarts"],
    packages=["smarts_imitation", "smarts_imitation.utils"]
)
