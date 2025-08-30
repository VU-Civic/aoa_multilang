from setuptools import setup, find_packages
setup(
    name='aoa_multilang',
    version='0.1.0',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    install_requires=['numpy', 'scipy', 'click', 'soundfile', 'matplotlib'],
    entry_points={'console_scripts': ['aoa-sim=aoa_sandbox.cli:cli']}
)
