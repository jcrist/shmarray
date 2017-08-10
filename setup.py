from os.path import exists
from setuptools import setup

packages = ['shmarray']
version = '0.0.1'

setup(name='shmarray',
      version='0.0.1',
      description='NumPy arrays backed by shared memory.',
      url='http://github.com/jcrist/shmarray',
      maintainer='Jim Crist',
      license='BSD',
      packages=packages,
      long_description=(open('README.rst').read() if exists('README.rst')
                        else ''),
      zip_safe=False)
