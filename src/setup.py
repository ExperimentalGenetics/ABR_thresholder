from distutils.core import setup

setup(
    name='ABRThresholdDetection',
    version='0.1.0',
    packages=['ABR_ThresholdFinder_NN', 'ABR_ThresholdFinder_SLR'],
    install_requires=['jupyterlab', 'pandas', 'statsmodels', 'scipy', 'scikit-learn', 'autopep8', 'matplotlib', 'seaborn', 'progressbar2', 'pingouin', 'more-itertools', 'tensorflow==1.15'],
    long_description=open('../README.md').read(),
)
