import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pystrand",
    version="0.0.1",
    author="Jiri Podivin",
    author_email="jpodivin@gmail.com",
    description="Python genetic algorithm package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jpodivin/pystrand",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Life",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy"
    ],
    extras_require={
        "tests": [
            "pytest",
            "flake8"
        ],
    },
)
