# Automatically parse the requirements.txt file for project requirements
def parse_requirements(filename):
    ''' Load requirements from a pip requirements file '''
    with open(filename, 'r') as fd:
        lines = []
        for line in fd:
            line.strip()
            if line and not line.startswith("#"):
                lines.append(line)
    return lines

requirements = parse_requirements('requirements.txt')
print(requirements)

# Load version from _version.py
version = {}
with open("signalforge/_version.py") as f:
    exec(f.read(), version)

from setuptools import setup, find_packages
setup(name='SignalForge',
    version=version["__version__"],
    license='MIT license',
    author='Giulio Curti',
    author_email='giulio.curti@dottorandi.unipg.it',
    description='Tools for analyzing and generating non-Gaussian, non-stationary signals in engineering applications.',
    url='https://github.com/GiulioCurti/SignalForge',
    packages=find_packages(),
    install_requires=requirements,
    )