#include <limits>
#include <map>
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <sstream>
#include <cmath>
#include "chrono"

#define DATASET_PATH "dataset.csv"
#define PRINT_FORMAT "Accuracy: %.2f%%\n"

struct stats {
    double mean;
    double std;
};

typedef struct stats stats;

double runDataProcessing(std::string, double);
