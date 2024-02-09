from setuptools import setup, find_packages

setup(
    name='odc',
    version='0.0.1',
    description='This project contains functions used to perform spatial data analysis. It is mainly focused on Mexican cities but can be used for different areas around the world.',
    author='Observatorio de Ciudades',
    author_email='observatoriodeciudades.tec@gmail.com',
    url='https://github.com/Observatorio-Ciudades/odc',
    packages=find_packages(),
    install_requires=[
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
    ],
)
