from setuptools import find_packages, setup

with open("requirements.txt") as f:
    required = f.read().splitlines()


setup(
    name="src",
    packages=find_packages(),
    version="0.1.0",
    description="Building furniture with diffusion",
    author="ankile",
    license="MIT",
    install_requires=required,
)
