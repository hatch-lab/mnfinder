import setuptools

with open("README.md", "r") as fh:
  long_description = fh.read()

setuptools.setup(
  name="mnfinder",
  version="1.0.0",
  author="Lucian DiPeso",
  author_email="ldipeso@uw.edu",
  description="A collection of U-Net micronucleus segmentation neural nets",
  long_description=long_description,
  long_description_content_type="text/markdown",
  url="https://github.com/hatch-lab/mnfinder",
  packages=setuptools.find_packages(),
  classifiers=(
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
  ),
)