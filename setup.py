from setuptools import setup, find_packages

requirements_file = open("test_requirements.txt")
test_requirements_file = open("test_requirements.txt")

setup(
    name="behavior_cloning",
    description="Solutions for CarND Behavioral Cloning project",
    url="https://github.com/daniel-m-campos/Behavioral-Cloning-Project",
    author="Daniel Campos",
    version="0.0.1",
    package_dir={"": "src"},
    packages=find_packages("src"),
    install_requires=requirements_file.read().strip().split("\n"),
    extras_require=test_requirements_file.read().strip().split("\n"),
)
