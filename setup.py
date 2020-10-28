
# -*- coding: utf-8 -*-

# DO NOT EDIT THIS FILE!
# This file has been autogenerated by dephell <3
# https://github.com/dephell/dephell

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

readme = ''

setup(
    long_description=readme,
    name='ideal-pancake',
    version='0.1.0',
    python_requires='==3.*,>=3.7.0',
    author='s0lvang',
    author_email='august.s.solvang@gmail.com',
    license='MIT',
    packages=['datasets.emip_dataset.deep_em_classifier.sp_tool', 'datasets.emip_dataset.deep_em_classifier.sp_tool.build.lib.sp_tool', 'datasets.emip_dataset.deep_em_classifier.sp_tool.build.lib.sp_tool.examples', 'datasets.emip_dataset.deep_em_classifier.sp_tool.examples', 'trainer'],
    package_dir={"": "."},
    package_data={"datasets.emip_dataset.deep_em_classifier.sp_tool": ["*.json", "*.md", ".git/hooks/*.sample", ".git/objects/pack/*.idx", ".git/objects/pack/*.pack", "baselines/*.csv", "figures/*.png", "test_data/*.arff", "test_data/*.asc", "test_data/*.coord", "test_data/*.edf", "test_data/*.txt"]},
    install_requires=['cloudml-hypertune', 'google-api-python-client', 'google-cloud-storage==1.*,>=1.32.0', 'matplotlib==3.*,>=3.3.2', 'numba==0.*,>=0.51.2', 'numpy==1.*,>=1.19.2', 'pandas==1.*,>=1.1.2', 'pandas-gbq==0.14.0', 'pyedflib==0.*,>=0.1.18', 'scikit-learn==0.*,>=0.23.2', 'scikit-plot', 'six==1.*,>=1.13.0', 'sktime==0.*,>=0.4.3', 'tensorflow==2.2.0', 'tsfresh==0.*,>=0.17.0', 'xlrd==1.*,>=1.2.0'],
    extras_require={"dev": ["black", "jupyter==1.*,>=1.0.0"]},
)
