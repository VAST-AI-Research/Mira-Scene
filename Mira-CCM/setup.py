from setuptools import find_packages, setup

setup(
    name="mira-ccm",
    version="1.0.0",
    description="Mira-CCM for training and inference diffusion models",
    package_dir={"": "src"},
    packages=find_packages("src"),
    python_requires=">=3.8",
    long_description="Mira-CCM training and inference diffusion models.",
    long_description_content_type="text/plain",
)
