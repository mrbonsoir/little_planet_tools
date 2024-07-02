# ReadMe

**2nd July 2024**

I have added the following:
+ new module to create config *.pto* file
+ test image
+ python notebook with exemple of usage of the module

**28 June 2024**

Since I started this repo *little planet tools* to create and manipulate little planet images, I changed computer, Python has grew up, newest modules are now available, so it's time for an update.

# What do we here?

We do the same as before, we use [hugin][hugin-link] and its sciptability in order to not use its graphical user interface.

The developped tool here is simply generating a config file *.pto* that can be called by the program *nona* in command line to perform image transformation from an equirectangular image representation of a 360º image panorama to an stereographic image that - with the right parameters - can produce a little planet.

The code can be used as an external python module, and I'm sharing an example of usage with Python notebook.

Also noticed that the code has been developped for use on a Macbook Pro M2 computer. Therefore for the code is pointing toward where hugin is installed on your machine.

# What do you need to use this code?

## Installation of Hugin on Macbook Pro M2

Visit this link https://groups.google.com/g/hugin-ptx/c/UbBpARzR3b8 and select *build with official 2023 code here(without gpu fix): https://bitbucket.org/Dannephoto/hugin/downloads/Hugin-2023.0.0.dmg*

The first install I have done was semi successfull as the program was crashing after each attempt of launching hugin.

## Python and virtual environment

Info about virtualenv [here][link-virtual]

## Imagemagick

It is not necessary but I'm using this list of image tools manipulation in command line a lot.

More info [here][link-imagemagick].

[link-hugin]: http://hugin.sourceforge.net/
[link-imagemagick]: http://www.imagemagick.org/
[link-virtual]:https://virtualenv.pypa.io/en/latest/installation.html