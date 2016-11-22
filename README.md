# Innopolis 2016 Data Mining Paper
The implementation of classifier proposed in paper considered as semester project on Innopolis Data Mining course.

## Kingfisher
Unfortunately kingfisher algorithm implementation is not working under linux or MacOS, it works only in Windows OS.
To compile and run kingfisher do the following:

1. Download and install [CMake](https://cmake.org/download/) for Windows

2. Download and install [MinGW](https://sourceforge.net/projects/mingw/files/) for Windows

3. Ensure that MinGW/bin and CMake/bin folders are in your PATH variable

4. Open Command Line in Kingfisher folder
 
5. `cmake -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=RELEASE -DBUILD_SHARED_LIBS=0 ./`

6. `mingw32-make`

8.  You can run rule generation by command with such format `kingfisher.exe -i <train dataset path> -k125 -w1 -M0.05 -g<class labels separated by '-'> -c<min conf> -m0.1 -o<output file>`.

    For example `kingfisher.exe -i ../splits/adult/train1.txt -k125 -w1 -M0.05 -g96-97 -c0.7 -m0.1 -o../rules/adult/train1_rules.txt`

## Classification
We provide already generated rules for each dataset split (from __splits__ folder) in __rules__ folder.

You need [NumPy](http://www.numpy.org/) to be installed (you can type `pip install numpy` to install it).

The implementation of proposed classifier is placed in file 'pos_neg_rules_classifier.py'.

To run classification you need to type `python evaluation.py`. 
It will train classifier several times on rules from __rules__ folder using corresponding datasets from __splits__ folder 
(with names of each split in format 'train#.txt'). 
And for each trained instance it will evaluate it on corresponding test dataset splits from __splits__ folder 
(for example, for 'train1_rules.txt' and 'train1.txt' it will use 'test1.txt' as test set). 

As a result it will produce file 'reduce.csv' with mean accuracy for each dataset on different method of classification