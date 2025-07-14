from setuptools import setup, find_packages

setup(
    name="O1NumHess",
    version="0.1.1",
    packages=find_packages(),
    install_requires=["numpy","scipy"],
    python_requires=">=3.6",
)
