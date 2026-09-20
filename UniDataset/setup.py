from setuptools import find_packages, setup

setup(
    name="UniDataset",
    version="1.0.0",
    description="Minimal UniDataset for Mira-Scene",
    package_dir={"": "src"},
    packages=find_packages("src"),
    python_requires=">=3.8",
    long_description="Minimal UniDataset subset for Mira-Scene training.",
    long_description_content_type="text/plain",
)
