#!/usr/bin/bash

sudo rm /usr/bin/python
sudo ln -s /usr/bin/python3 /usr/bin/python

# Needed for ray
sudo apt-get install rsync

# Could run this script directly using curl and the github raw link
git clone https://github.com/ca-scribner/quick-redraw.git

pip3 install -r quick-redraw/requirements_dev.txt

# RUN WHAT I WANT