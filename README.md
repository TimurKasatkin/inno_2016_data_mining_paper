# Innopolis 2016 Data Mining Paper
The implementation of classifier proposed in paper considered as a semester project on Innopolis Data Mining course.

## Datasets
We have split all datasets into 10 parts each for cross validation and written them into files with name in format:

- **train#.txt** - train part with number #

- **test#.txt** - test part with number #

They are in folder __split__ inside folders, named as corresponding datasets. 
For example, file **train1.txt** in folder **adult** is a 1st fold of _adult_ dataset.

## Kingfisher
Unfortunately, kingfisher algorithm's implementation doesn't work under linux or MacOS, it works only on Windows OS.
To compile and run kingfisher do the following:

1. Download and install [CMake](https://cmake.org/download/) for Windows

2. Download and install [MinGW](https://sourceforge.net/projects/mingw/files/) for Windows

3. Ensure that MinGW/bin and CMake/bin folders are in your PATH variable

4. Open Command Line in **kingfisher** folder
 
5. `cmake -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=RELEASE -DBUILD_SHARED_LIBS=0 ./`

6. `mingw32-make`

8.  You can run rule generation by command with such format `kingfisher.exe -i <train dataset path> -k125 -w1 -M0.05 -g<class labels separated by '-'> -c0.7 -m0.1 -o<output file>`.

    For example `kingfisher.exe -i ../splits/adult/train1.txt -k125 -w1 -M0.05 -g96-97 -c0.7 -m0.1 -o../rules/adult/train1_rules.txt`  

## CARs
We provide already generated rules for each dataset fold in __rules__ folder.

## Classification

You need [NumPy](http://www.numpy.org/) to be installed (you can type `pip install numpy` to install it).

The implementation of proposed classifier is placed in file 'classifier.py'.

To run classification you need to type `python evaluation.py`. 
It will train classifier several times on rules from __rules__ folder using corresponding datasets from __splits__ folder. 
And for each trained instance it will evaluate it on corresponding test dataset's split from __splits__ folder 
(for example, instance trained on **train1_rules.txt** and **train1.txt** will be evaluated using **test1.txt**). 

As a result it will produce file **result.csv** with mean accuracy for each dataset on different methods of assigning class:
- BEST - rule with highest confidence
- POS_SUM - only positive rules with group of highest sum of confidences
- POS_AVE - only positive rules with group of highest mean confidence
- BOTH_SUM - both types of rules with group of highest sum of confidences
- BOTH_AVE - both types of rules with group of highest mean confidence