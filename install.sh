# Build Environment 
mkdir tmp 
mkdir exp

# Install Mockturtle 
cd src/mockturtle 
mkdir build
cd build
cmake ..
make my_baseline 
make my_mapper 

# Install Kissat 
cd ../../kissat 
./configure
cd build 
make

