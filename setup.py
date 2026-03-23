from setuptools import find_packages, setup


setup(
    name="metric-matching",
    version="0.1.0",
    description="Lightning implementation of Riemannian Metric Matching on 3dshapes",
    package_dir={"": "src"},
    packages=find_packages("src"),
    install_requires=[
        "torch>=2.2",
        "lightning>=2.2",
        "wandb>=0.17",
        "h5py>=3.10",
        "numpy>=1.24",
        "pillow>=10.0",
    ],
    extras_require={
        "dev": [
            "pytest>=8.0",
        ]
    },
)
