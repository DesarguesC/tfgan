from setuptools import setup, find_packages

setup(
    name='tfgan',
    version='0.0.1',
    description='',
    author='none',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'tqdm',
        'pillow',
        'matplotlib',
    ],
)


