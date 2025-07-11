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

with open('.\SignalForge\__init__.py', 'r', encoding='utf-8') as file:
    for line in file:
        line = line.strip()
        if line.startswith("__version__ ="):
            # Extract the version string between single quotes
            start = line.find("'")
            end = line.rfind("'")
            if start != -1 and end != -1 and end > start:
                __version__ =  line[start+1:end]
                break


from setuptools import setup, find_packages
setup(name='SignalForge',
    version=__version__,
    license='MIT license',
    author='Giulio Curti',
    author_email='giulio.curti@dottorandi.unipg.it',
    description='Tools for analyzing and generating non-Gaussian, non-stationary signals in engineering applications.',
    url='https://github.com/GiulioCurti/SignalForge',
    packages=find_packages(),
    install_requires=requirements,
    )