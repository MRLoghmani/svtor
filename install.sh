# SET UP FOR OPENCV AND SKLEARN IN PYTHON 3.4
# Author: Pranshu Gupta
#######################################################################################

# Installing scipy stack python3
sudo apt-get install build-essential python3-dev python3-setuptools python3-numpy python3-scipy libatlas-dev libatlas3gf-base
sudo update-alternatives --set libblas.so.3 /usr/lib/atlas-base/atlas/libblas.so.3
sudo update-alternatives --set liblapack.so.3 /usr/lib/atlas-base/atlas/liblapack.so.3

# Installing matplotlib python3
sudo apt-get install python3-matplotlib

# Installing pip3 for python3
sudo apt-get install python3-pip

# Installing sklearn python3
sudo pip3 install scikit-learn

sudo apt-get update
sudo apt-get upgrade

# Installing cmake with python3
sudo apt-get install build-essential cmake git pkg-config

# Installing some useful libraries
sudo apt-get install libjpeg8-dev libtiff4-dev libjasper-dev libpng12-dev
sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
sudo apt-get install libgtk2.0-dev
sudo apt-get install libatlas-base-dev gfortran

# Installing opencv with python3
git clone https://github.com/Itseez/opencv.git
cd opencv
git checkout 3.1.0
git clone https://github.com/Itseez/opencv_contrib.git
cd opencv_contrib
git checkout 3.1.0

# cmake 
cd ~/opencv
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D INSTALL_C_EXAMPLES=OFF -D INSTALL_PYTHON_EXAMPLES=ON -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules -D BUILD_EXAMPLES=ON -D PYTHON_EXECUTABLE=/usr/bin/python3.4 -D PYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.4m.so -D PYTHON_INCLUDE_DIR=/usr/include/python3.4m -DPYTHON_INCLUDE_DIR2=/usr/include/x86_64-linux-gnu/python3.4m -D PYTHON_NUMPY_INCLUDE_DIRS=/usr/lib/python3/dist-packages/numpy/core/include/ ..
make -j4

sudo make install