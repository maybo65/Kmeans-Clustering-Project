from setuptools import setup, Extension

setup(
    name='Interface',
    version='0.1.0',
    ext_modules=[Extension('Interface', ['kmeans.c','algebra.c','print.c','Interface.c'])]
)