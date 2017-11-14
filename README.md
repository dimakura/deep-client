# Setup instructions

You will need Python 3.6 for running example code.

**Note**: MacOS usually comes with Python 2.7 preinstalled, but this is not sufficient.

To check presence of Python 3 on your system:

```sh
python3 -V
# Python 3.6.2
```

If you don't have Python 3 yet, use `brew` for installation:

```sh
brew install python3
```

Also install standard package manager for Python:

```sh
sudo easy_install pip
```

All you need to do after this is to clone this repository and run setup script.

To clone this repository to your computer:

```sh
git clone https://github.com/dimakura/deep-client.git
cd deep-client/
```

And run the following scripts from project's home directory:

```sh
export VIRTUALENV_PATH=.env

pip3 install virtualenv
virtualenv ${VIRTUALENV_PATH}
source ${VIRTUALENV_PATH}/bin/activate

pip3 install numpy
pip3 install http://download.pytorch.org/whl/torch-0.2.0.post3-cp36-cp36m-macosx_10_7_x86_64.whl
pip3 install torchvision
```
