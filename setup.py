from setuptools import setup, find_packages

setup(
    name="match-series-batch",
    version="0.1.0",
    author="Haoran Ma",
    author_email="haoran.ma@ikz-berlin.de",
    description="Batch non-rigid image alignment using pymatchseries.",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/HaoranLMaoMao/match-series-batch",
    packages=find_packages(),
    install_requires=[
        "hyperspy>=1.6.1",
        "pymatchseries>=0.1.0",
        "tqdm",
        "Pillow",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    entry_points={
        'console_scripts': [
            'match-series-batch=match_series_batch.main:main',
        ],
    },
)
