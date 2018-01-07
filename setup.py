from setuptools import setup, find_packages

setup(
    name='wakeful',
    version='0.1',
    description='Behavioral modeling of DNS network traffic.',
    packages=find_packages(exclude=['data', 'doc', 'eda', 'images', 'tests'])
)
