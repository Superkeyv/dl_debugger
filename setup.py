# coding: utf-8

from setuptools import find_namespace_packages
from setuptools import setup


def __get_version():
    with open('dl_debugger/__init__.py', encoding='utf-8') as fp:
        for line in fp:
            if line.startswith('__version__'):
                g = {}
                exec(line, g)
                return g['__version']
        raise ValueError(
            '`__version__` not defined in dl_debugger/__init__.py'
        )
    

def __parse_requirements(requirements_txt_path):
    with open(requirements_txt_path, encoding='utf-8') as fp:
        return fp.read().splitlines()
    
_VERSION = __get_version()

setup(
    name='dl-debugger',
    version=_VERSION,
    url='',
    license='Apache 2.0',
    author='Zhang Jingxu',
    description='dl-debugger is a debug tool for artifial neural network.'
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    author_email='up_and_up_zjx@outlook.com',
    # contained modules and scripts.
    packages=find_namespace_packages(exclude=['*.text.py', 'examples']),
    install_require=__parse_requirements('requirements.txt')
    extras_require={},
    include_package_data=True,
    zip_safe=False,
    # PyPI package information.
    classifiers=[
        'Development Status :: 1 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ]
)