#include <limits>
#include <map>
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <sstream>
#include <cmath>
#include "chrono"
#include <semaphore.h>
#include <thread>

#define PRINT_FORMAT "Accuracy: %.2f%%\n"
#define NUM_OF_THREADS 4
#define CSV ".csv"
#define DATASET_PATH "dataset_"

struct stats {
    double mean;
    double std;
};

typedef struct stats stats;

double runDataProcessing(std::string, double);
