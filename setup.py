import io
from setuptools import find_packages, setup
setup(
    name             = 'failure-clustering-engine',
    version          = '1.0',
    description      = 'An engine for failure clustering',
    author           = 'Gabin An',
    author_email     = 'agb94@kaist.ac.kr',
    url              = 'https://github.com/Suresoft-GLaDOS/failure_clustering',
    download_url     = 'https://github.com/Suresoft-GLaDOS/failure_clustering',
    install_requires = ['numpy==1.20.2', 'scikit-learn==0.24.2', 'scipy==1.6.3'],
    packages         = find_packages(),
    keywords         = ['failure clustering'],
    python_requires  = '>=3.8',
    package_data={},
    zip_safe=True,
)
