from setuptools import setup, find_packages

setup(
    name="coda-tools",
    version="0.1.0",
    description="Tools for working with CODa",
    author="Dongmyeong Lee",
    author_email="domlee@utexas.edu",
    packages=find_packages(exclude=["scripts"]),
    install_requires=[
        "numpy",
        "scipy",
        "pyyaml",
        "sckikit-learn",
        "matplotlib",
        "pandas",
        "tqdm",
        "pyyaml",
        "black",
        "open3d",
        "rospkg",
    ],
)
