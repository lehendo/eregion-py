from setuptools import setup, find_packages

setup(
    name="eregion",
    version="0.1.0",
    description="A comprehensive, accessible neural network analytics library for PyTorch and TensorFlow.",
    author="Pratyay Pandey",
    author_email="p.pandey@berkeley.edu",
    url="https://github.com/BitLegion/eregion",
    packages=find_packages(),
    install_requires=[
        'torch',
        'tensorflow',
        'numpy',
        'requests',
        'scikit-learn',
        'matplotlib',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)
