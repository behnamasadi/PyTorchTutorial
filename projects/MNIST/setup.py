from setuptools import setup, find_packages

setup(
    name="mnist_project",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'torch>=2.0.0',
        'torchvision>=0.15.0',
        'numpy>=1.21.0',
        'PyYAML>=6.0',
        'tqdm>=4.65.0',
        'matplotlib>=3.5.0',
    ],
)
