from setuptools import setup, find_packages

setup(
    name='bpm-prediction-project',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A project for predicting the beats per minute of songs using machine learning.',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'pandas',
        'numpy',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'jupyter'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)