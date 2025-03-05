#pragma once

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define __CUDA_INTERNAL_COMPILATION__
#include <math_functions.h>
#undef __CUDA_INTERNAL_COMPILATION__

#include <iostream>
#include <stdio.h>
#include <cmath>
#include <algorithm>
#include <utility>
#include <string>
#include <ctime>
#include <stdlib.h>
#include <time.h>
#include <fstream>
#include <vector>
#include <curand_kernel.h>