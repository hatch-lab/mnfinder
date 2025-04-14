---
title: "Emily's guide to running MNFinder for users new to python (like me) - MacOS"
author: "Emily Hatch"
date: "2025-04-11"
output: 
  html_document:
    keep_md: TRUE
---


## Input Images
MNFinder is designed to work on single section (or single projection) fluorescent images of DNA in adherent cultured cells. Both single channel and multichannel images can be used as input. MNFinder has been run successfully on images with micronucleus (MN) frequencies as low as 1% MN positive cells and as high as 70%.

For best results, imaged cells should be in a single layer, with a confluency up to 80%. 

MNFinder can take any image resolution or image size as input. However, prior to analysis, images should be resized to a scale between 1.55 - 2.8 px/um (i.e. shorter nuclei diameter ~ 30 pixels). 

### Install analysis templates and mnfinder via terminal/shell
1. On your computer, make a folder for all of your mnfinder analysis work.

2. From the main mnfinder page (https://github.com/hatch-lab/mnfinder/tree/main), click on mnfinder-test.ipynb and download it (the "raw" file) to the folder you just made.

3. Do the same for mnfinder-batch.ipynb.

4. Go to https://github.com/hatch-lab/mnfinder/src/mnfinder/training-data/2022-04-14_RPE1/images/5.tif and download this image to your mnfinder directory. You will use this image to test your installation of mnfinder.

5. Open terminal. Haven't used it in awhile? Here's a good cheat sheet https://whatbox.ca/wiki/Bash_Shell_Commands. Need help getting started with shell? Fred Hutch lists some good resources here: https://sciwiki.fredhutch.org/scicomputing/software_linux101/

6. Navigate to your mnfinder folder using the cd command.

7. Check your version of python.

```
python

```

  The first line lists your default python version. mnfinder requires      python version 3.12.X. If you have that, great! If you have an earlier   version install python 3.12 using homebrew.

````
# Escape the previous menu by hitting "Ctrl-d"
# Install homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

````
  Install python 3.12

````
brew install python@3.12

````
  You may find that python 3.12 is already installed, but not the   default.   This is fine, keep going! 

8. Create a virtual environment using python 3.12 to run mnfinder in.

````
python3.12 -m venv .env  

````
9. Activate your new virtual environment.

````
source .env/bin/activate

````
10. Install mnfinder

````
pip install mnfinder

````

11. Once mnfinder is installed launch Jupyter Notebook in a browser tab. This will allow you to use the two .ipynb files you downloaded earlier to run mnfinder. Start with mnfinder_test.ipynb. It will guide you on your first mnfinder analysis.

````
jupyter lab

````
#### Notes on using Jupyter notebooks.
1. Open mnfinder_test.ipynb by double-clicking on it.

2. Follow the directions in this notebook.

3. Run a single code block in the notebook by highlighting it and hitting play. When the code is running, you'll see an asterix next to the block. When it's done you'll see a number.

4. When you are done with your analysis, close the jupyter notebook tab in your browser. Then press Ctrl+c in terminal and y to end the program. 

5. Type exit to close terminal. 

### How to run mnfinder the second time
1. Good news! It's a lot simpler and quicker.

2. Open terminal and navigate to your mnfinder folder using the cd command.

3. Activate the python 3.12 virtual environment you previously set up

````
source .env/bin/activate

````
4. Launch jupyter notebook
````
jupyter lab

````
