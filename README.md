# Setup instructions

## 1. Python3

You will need Python 3.6 for running example code.

To check presence of Python 3 on your system:

```sh
python3 -V
# Python 3.6.2
```

### MacOS

MacOS usually comes with Python 2.7 preinstalled, but this is not sufficient.

If you don't have Python 3 yet, use `brew` for installation:

```sh
brew install python3
```

Also install standard package manager for Python:

```sh
sudo easy_install pip3
```

### Ubuntu

Ubuntu 16.04 and later come with both Python 2 and 3 preinstalled.

```sh
sudo apt-get update
sudo apt-get -y upgrade
```

Make sure you have pip3 installed as well:

```sh
sudo apt-get install -y python3-pip
```

## 2. Clone deep-client repo

The next step is to clone `deep-client` repository to your computer:

```sh
git clone https://github.com/dimakura/deep-client.git
cd deep-client/
```

## 3. Environment setup

We use [virtualenv](https://virtualenv.pypa.io/en/stable/) to manage working environment.

To create a new environment:

```sh
pip3 install virtualenv
virtualenv .env
```

All the remaining steps should be done inside this new environment.
To activate environment, issue the following command:

```sh
source .env/bin/activate
```

## 4. Install PyTorch

Please refer to [PyTorch](http://pytorch.org/) installation page in case of trouble.

### MacOS

```sh
pip3 install http://download.pytorch.org/whl/torch-0.2.0.post3-cp36-cp36m-macosx_10_7_x86_64.whl
```

### Ubuntu

```sh
pip3 install http://download.pytorch.org/whl/cu75/torch-0.2.0.post3-cp36-cp36m-manylinux1_x86_64.whl
```

## 5. Additional libraries

Install all other libraries using pip:

```sh
pip3 install torchvision
pip3 install numpy
pip3 install jupyter
```

## Working with environment

To activate working environment:

```sh
source .env/bin/activate
```

When you are done working with it, issue:

```sh
deactivate
```
