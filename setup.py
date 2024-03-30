from setuptools import setup, find_packages

setup(
    name='semantic_similarity',
    version='0.3.1',
    packages=find_packages(),
    package_dir={"":"."},
    include_package_data=True,
    install_requires=[
        "pronto",
        "numpy",
        "pandas",
        "scipy",
        "requests",
        "omegaconf"
    ],
)
