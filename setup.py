from setuptools import setup, find_packages

setup(
    name='wakeful',
    version='1.0',
    url='https://github.com/robOcity/wakeful',
    description='Behavioral modeling of DNS network traffic.',
    packages=find_packages(exclude=['data', 'doc', 'images', 'test']),
    author='Rob Osterburg',
    author_email='robert.osterburg@gmail.com'
)
