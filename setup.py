from setuptools import setup, find_packages


__version__ = '0.0.1'

setup(
    name='HomoFusion',
    version=__version__,
    author='Shan Wang',
    author_email='shan.wang@anu.edu.au',
    url='https://github.com/ShanWang-Shan/HomoFusion',
    license='CSIRO OSS Non Commercial License',
    packages=find_packages(include=['homo_transformer', 'homo_transformer.*']),
    zip_safe=False,
)
