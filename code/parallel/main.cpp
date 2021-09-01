#include "dataProcessor.hpp"

int main(int argc, char *argv[]) {
    runDataProcessing(argv[1], std::stod(argv[2]));
    return 0;
}