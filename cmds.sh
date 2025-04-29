rm -rf build
mkdir build
cd build

cmake .. -DCUDA_AVAILABLE=OFF
make



./bin/sqlqueryprocessor