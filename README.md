I-SEA: Importance Sampling and Expected Alignment-based Deep Distance Metric Learning for Time Series Analysis and Embedding
===============


The codebase is written in python, C, and MATLAB.

The MATLAB-based baselines are under `TimeSeriesMetricLearning/matlab`.
The python (learning-based) baselines are under `TimeSeriesMetricLearning/python_pytorch`, and the following are the commands to compile the C-based libraries for I-SEA/DECADE. 

- DECADE
```
gcc -c -fPIC edist_c_path_py.c -o edist_c_path_py.o
gcc edist_c_path_py.o -shared -o edist_c_path.so
```
- DECADE IS
```
gcc -c -fPIC edist_is_c_path_py.c -o edist_is_c_path_py.o
gcc edist_is_c_path_py.o -shared -o edist_is_c_path.so

gcc -c -fPIC -I/usr/include edist_is_c_path_py.c -o edist_is_c_path_py.o
gcc edist_is_c_path_py.o -shared -o edist_is_c_path.so -L/usr/lib/x86_64-linux-gnu -lgsl -lgslcblas -lm
```
- DECADE IS (KDE)
```
gcc -c -fPIC -I/usr/include kde-edist_is_c_path_py.c -o kde-edist_is_c_path_py.o
gcc kde-edist_is_c_path_py.o -shared -o kde-edist_is_c_path_py.so -L/usr/lib/x86_64-linux-gnu -lgsl -lgslcblas -lm
```
