# inno_2016_data_mining_paper
The implementation of classifier proposed in paper considered as semester project on Innopolis Data Mining course.

##Kingfisher
Unfortunatly kingfisher algorithm implementation is not working under linux or MacOS, it works only in Windows OS.
To compile and run kingfisher do the following:
1. Download and install [CMake](https://cmake.org/download/) for Windows
2. Download and install [MinGW](https://sourceforge.net/projects/mingw/files/) for Windows
3. Ensure that MinGW/bin and CMake/bin folders are in your
4. Open Command Line in Kingfisher folder 
5. cmake -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=RELEASE -DBUILD_SHARED_LIBS=0 ./
6. mingw32-make
7. kingfisher.exe -i <discretized dataset path> -k125 -M-0.1 -g<class labels> -c<min conf> -m0.1 -o<output file> 