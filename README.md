# Setup instructions

## 1. Python3

You will need Python 3.6 for running example code.

To check presence of Python 3 on your system:

```sh
$ python3 -V
# Python 3.6.3
```

### MacOS

MacOS usually comes with Python 2.7 preinstalled, but this is not sufficient.

If you don't have Python 3 yet, use `brew` for installation:

```sh
$ brew install python3
```

Also install pip3 and virtualenv:

```sh
$ sudo easy_install pip
$ pip install --upgrade virtualenv
```

### Ubuntu

Ubuntu 16.04 and later come with both Python 2 and 3 preinstalled.

Additionally install pip package manager and virtualenv:

```sh
$ sudo apt-get install python3-pip python3-dev python-virtualenv
```

## 2. Clone deep-client repo

The next step is to clone `deep-client` repository to your computer:

```sh
$ git clone https://github.com/dimakura/deep-client.git
$ cd deep-client/
```

## 3. Environment setup

To create a new environment:

```sh
$ virtualenv --system-site-packages -p python3 .env
```

All the remaining steps should be done inside this new environment.
To activate environment, issue the following command:

```sh
$ source .env/bin/activate
```

This should change command prompt to:

```sh
(.env) $
```

Ensure pip is installed:

```sh
(.env) $ easy_install -U pip
```

## 4. Install PyTorch

Please refer to [PyTorch](http://pytorch.org/) installation page in case of trouble.

### MacOS

```sh
(.env) $ pip3 install http://download.pytorch.org/whl/torch-0.2.0.post3-cp36-cp36m-macosx_10_7_x86_64.whl
```

### Ubuntu

```sh
(.env) $ pip3 install http://download.pytorch.org/whl/cu75/torch-0.2.0.post3-cp36-cp36m-manylinux1_x86_64.whl
```

## 5. Additional libraries

Install all other libraries using pip:

```sh
(.env) $ pip3 install torchvision
(.env) $ pip3 install --upgrade jupyter
```

## Working with environment

To activate working environment:

```sh
$ source .env/bin/activate
(.env) $
```

You can quickly test PyTorch installation with:

```sh
(.env) $ python3 -c "import torch; print(torch.Tensor([1]))"
#
#  1
# [torch.FloatTensor of size 1]
```

When you are done working with it, issue:

```sh
(.env) $ deactivate
$
```

## Working with Jupyter notebooks

To start jupyter notebooks:

```sh
(.env) $ jupyter notebook
```

and navigate to `notebooks/` folder.
