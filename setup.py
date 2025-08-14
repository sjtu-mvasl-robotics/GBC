from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

setup(
    name="GBC",
    version="1.0.1",
    author="Yifei Yao",
    author_email="yifyao@live.cn",
    description="A package for GBC",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sjtu-mvasl-robotics/GBC.git",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    install_requires=requirements,
    python_requires=">=3.8",

    # entry_points={
    #     "console_scripts": [
    #         "gbc = gbc.cli:main",
    #     ],
    # },

    include_package_data=True,
)
