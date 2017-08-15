from os.path import exists
from setuptools import setup
from setuptools.extension import Extension

packages = ['shmarray']
version = '0.0.1'


extensions = [Extension("shmarray.shmbuffer",
                        ['shmarray/shmbuffer.c'])
              ]

setup(name='shmarray',
      version='0.0.1',
      description='NumPy arrays backed by shared memory.',
      url='http://github.com/jcrist/shmarray',
      maintainer='Jim Crist',
      license='BSD',
      packages=packages,
      long_description=(open('README.rst').read() if exists('README.rst')
                        else ''),
      ext_modules=extensions,
      zip_safe=False)
