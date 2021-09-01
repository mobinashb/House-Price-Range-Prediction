#include "dataProcessor.hpp"
#include "chrono"

using namespace std::chrono;
using namespace std;

sem_t configSem;

struct DatasetConfig {
    vector<string> FIELD_NAMES;
    int NUM_OF_FIELDS;
    bool isSet = false;
    int NUM_OF_ROWS = 0;
    string TARGET_NAME;
    string SELECTED_FEATURE_NAME = "GrLivArea";
    double TARGET_THRESHOLD;
} DATASET_CONFIG;

void setDatasetConfig(string firstLine) {
    DATASET_CONFIG.isSet = true;
    string tmp;
    stringstream ss(firstLine);

    while (getline(ss, tmp, ',')) {
        DATASET_CONFIG.FIELD_NAMES.push_back(tmp);
    }
    DATASET_CONFIG.NUM_OF_FIELDS = DATASET_CONFIG.FIELD_NAMES.size();
}

void setDatasetSize(int size) {
    DATASET_CONFIG.NUM_OF_ROWS += size;
}

void setDatasetTargetName(string targetName) {
    DATASET_CONFIG.TARGET_NAME = targetName;
}

void setDatasetTargetThreshold(double threshold) {
    DATASET_CONFIG.TARGET_THRESHOLD = threshold;
}

struct DataThread {
    int threadNumber;
    thread dataThread;

    map<int, map<string, stats>> statsMap;
    map<int, int> numRows;
    vector<map<string, double>> dataset;
    vector<double> predictions;
    int countCorrect;
};

void readCSV(
    string filepath,
    vector<map<string,
    double>>& dataValues
) {
	std::fstream fin;
	fin.open(filepath, std::ios::in);
	map<string, double> row;
	std::string word, temp, firstLine;
    int numRows = 0;
	getline(fin, firstLine);
    sem_wait(&configSem);
    if (!DATASET_CONFIG.isSet) {
        setDatasetConfig(firstLine);
        setDatasetTargetName(DATASET_CONFIG.FIELD_NAMES[DATASET_CONFIG.NUM_OF_FIELDS - 1]);
    }
    sem_post(&configSem);
	while (fin >> temp) {
        numRows++;
		row.clear();
		std::stringstream s(temp);
        int fieldCounter = 0;
		while (getline(s, word, ',')) {
            row.insert(
                pair<string, double>(DATASET_CONFIG.FIELD_NAMES[fieldCounter],
                stod(word))
            );
            fieldCounter++;
        }
        // row.insert(
        //     pair<string, double>(DATASET_CONFIG.FIELD_NAMES[fieldCounter],
        //     stod(word))
        // );
        dataValues.push_back(row);
	}
	fin.close();
    setDatasetSize(numRows);
}

int getTargetLabel(double num) {
    int label = num >= DATASET_CONFIG.TARGET_THRESHOLD;
    return label;
}

void labelizeTarget(
    vector<map<string, double>>& dataset
) {
    vector<map<string, double>>::iterator vecItr;

    for (vecItr = dataset.begin(); vecItr != dataset.end(); ++vecItr) {
        (*vecItr)[DATASET_CONFIG.TARGET_NAME] = getTargetLabel((*vecItr)[DATASET_CONFIG.TARGET_NAME]);
    }
}

map<int, map<string, stats>> initializeStats() {
	map<int, map<string, stats>> statsMap;
    for (int i = 0; i < DATASET_CONFIG.NUM_OF_FIELDS; i++) {
        stats newStats;
        newStats.mean = 0;
        newStats.std = 0;
        statsMap.insert(
            make_pair(0, map<string, stats> {make_pair(DATASET_CONFIG.FIELD_NAMES[i], newStats)})
        );
        statsMap.insert(
            make_pair(1, map<string, stats> {make_pair(DATASET_CONFIG.FIELD_NAMES[i], newStats)})
        );
    }
    return statsMap;
}

void updateMeanSum(
    map<string, double>& current,
    map<int, map<string, stats>>& statsMap,
    map<int, int>& numRows
) {
    numRows[current[DATASET_CONFIG.TARGET_NAME]]++;
    for (int i = 0; i < DATASET_CONFIG.NUM_OF_FIELDS; i++) {
        statsMap[current[DATASET_CONFIG.TARGET_NAME]][DATASET_CONFIG.FIELD_NAMES[i]].mean +=
                current[DATASET_CONFIG.FIELD_NAMES[i]];
    }
}

void getMeanSum(
    vector<map<string, double>>& dataset,
    map<int, map<string, stats>>& statsMap,
    map<int, int>& numRows
) {
    numRows[0] = 0;
    numRows[1] = 0;
    vector<map<string, double>>::iterator vecItr;

    for (vecItr = dataset.begin(); vecItr != dataset.end(); ++vecItr) {
        updateMeanSum(*vecItr, statsMap, numRows);
    }
}

void updateMean(
    map<int, map<string, stats>>& statsMap,
    map<int, int> numRows
) {
    for (int i = 0; i < DATASET_CONFIG.NUM_OF_FIELDS; i++) {
        for (int targetVal = 0; targetVal <= 1; targetVal++)
            statsMap[targetVal][DATASET_CONFIG.FIELD_NAMES[i]].mean /= numRows[targetVal];
    }
}

void updateStdSum(
    map<string, double>& current,
    map<int, map<string, stats>>& statsMap
) {
    for (int i = 0; i < DATASET_CONFIG.NUM_OF_FIELDS; i++) {
        statsMap[current[DATASET_CONFIG.TARGET_NAME]][DATASET_CONFIG.FIELD_NAMES[i]].std +=
                pow(
                    current[DATASET_CONFIG.FIELD_NAMES[i]] -
                    statsMap[current[DATASET_CONFIG.TARGET_NAME]][DATASET_CONFIG.FIELD_NAMES[i]].mean, 2);
    }
}

void updateStd(
    map<int, map<string, stats>>& statsMap,
    map<int, int> numRows
) {
    for (int i = 0; i < DATASET_CONFIG.NUM_OF_FIELDS; i++) {
        for (int targetVal = 0; targetVal <= 1; targetVal++) {
            statsMap[targetVal][DATASET_CONFIG.FIELD_NAMES[i]].std /= numRows[targetVal];
            statsMap[targetVal][DATASET_CONFIG.FIELD_NAMES[i]].std =
                sqrt(statsMap[targetVal][DATASET_CONFIG.FIELD_NAMES[i]].std);
        }
    }
}

void getStdSum(
    vector<map<string, double>>& dataset,
    map<int, map<string, stats>>& statsMap
) {
    vector<map<string, double>>::iterator vecItr;

    for (vecItr = dataset.begin(); vecItr != dataset.end(); ++vecItr) {
        updateStdSum(*vecItr, statsMap);
    }
}

bool isInRange(double num, stats featureStats) {
    return (num >= (featureStats.mean - featureStats.std)) && (num <= (featureStats.mean + featureStats.std));
}

double predictTarget(
    map<string, double>& current,
    stats featureStats,
    string featureName
) {
    if (isInRange(current[featureName], featureStats))
        return 1;
    else return 0;
}

void predictAllTargets(
    vector<double>& predictions,
    vector<map<string, double>> dataset,
    map<int, map<string, stats>> statsMap
) {
    vector<map<string, double>>::iterator vecItr;
    for (vecItr = dataset.begin(); vecItr != dataset.end(); ++vecItr) {
        predictions.push_back(predictTarget(*vecItr,
            statsMap[1][DATASET_CONFIG.SELECTED_FEATURE_NAME], DATASET_CONFIG.SELECTED_FEATURE_NAME));
    }
}

void calculateCountCorrect(
    vector<double>& predictions,
    vector<map<string, double>>& dataset,
    int& countCorrect
) {
    int count = 0;
    int i = 0;
    vector<map<string, double>>::iterator vecItr;
    for (vecItr = dataset.begin(); vecItr != dataset.end(); ++vecItr) {
        if ((*vecItr)[DATASET_CONFIG.TARGET_NAME] == predictions[i])
            count++;
        i++;
    }
    countCorrect = count;
}

void mergeMeanVals(
    vector<DataThread*>& dataThreads,
    map<int, map<string, stats>>& statsMap,
    map<int, int>& numRows
) {
    statsMap = initializeStats();
    for (int fieldIter = 0; fieldIter < DATASET_CONFIG.NUM_OF_FIELDS; fieldIter++) {
        for (int threadIter = 0; threadIter < NUM_OF_THREADS; threadIter++) {
            for (int targetVal = 0; targetVal <= 1; targetVal++) {
                statsMap[targetVal][DATASET_CONFIG.FIELD_NAMES[fieldIter]].mean +=
                dataThreads[threadIter]->statsMap[targetVal][DATASET_CONFIG.FIELD_NAMES[fieldIter]].mean;
            }
        }
    }

    for (int threadIter = 0; threadIter < NUM_OF_THREADS; threadIter++) {
        for (int targetVal = 0; targetVal <= 1; targetVal++) {
            numRows[targetVal] += dataThreads[threadIter]->numRows[targetVal];
        }
    }
    updateMean(statsMap, numRows);
    for (int threadIter = 0; threadIter < NUM_OF_THREADS; threadIter++) {
        dataThreads[threadIter]->statsMap = statsMap;
    }
}

void mergeStdVals(
    vector<DataThread*>& dataThreads,
    map<int, map<string, stats>>& statsMap,
    const map<int, int> numRows
) {
    for (int fieldIter = 0; fieldIter < DATASET_CONFIG.NUM_OF_FIELDS; fieldIter++) {
        for (int threadIter = 0; threadIter < NUM_OF_THREADS; threadIter++) {
            for (int targetVal = 0; targetVal <= 1; targetVal++) {
                statsMap[targetVal][DATASET_CONFIG.FIELD_NAMES[fieldIter]].std +=
                (dataThreads[threadIter]->statsMap)[targetVal][DATASET_CONFIG.FIELD_NAMES[fieldIter]].std;
            }
        }
    }
    updateStd(statsMap, numRows);
    for (int threadIter = 0; threadIter < NUM_OF_THREADS; threadIter++) {
        dataThreads[threadIter]->statsMap = statsMap;
    }
}

void joinThreads(vector<DataThread*>& dataThreads) {
    for (int i = 0; i < NUM_OF_THREADS; i++) {
        dataThreads[i]->dataThread.join();
    }
}

void getThreadStdSum(DataThread* currentThread) {
    getStdSum(currentThread->dataset, currentThread->statsMap);
}

void getParallelInput(DataThread* currentThread, string directory) {
    string filename = directory + DATASET_PATH + to_string(currentThread->threadNumber) + CSV;
    readCSV(filename, currentThread->dataset);
    labelizeTarget(currentThread->dataset);
	currentThread->statsMap = initializeStats();
    getMeanSum(currentThread->dataset, currentThread->statsMap, currentThread->numRows);
}

void getParallelOutput(
    DataThread* currentThread,
    const map<int, map<string, stats>>& statsMap
) {
    predictAllTargets(currentThread->predictions, currentThread->dataset, statsMap);
    calculateCountCorrect(
        currentThread->predictions,
        currentThread->dataset,
        currentThread->countCorrect
    );
}

double calculateAccuracy(vector<DataThread*>& dataThreads) {
    int correctSum = 0;
    int totalNum = 0;
    for (int i = 0; i < NUM_OF_THREADS; i++) {
        correctSum += dataThreads[i]->countCorrect;
        totalNum += dataThreads[i]->dataset.size();
    }
    return (double)correctSum/ (double)totalNum;
}

double runDataProcessing(string directory, double threshold) {
    setDatasetTargetThreshold(threshold);

    auto totalStart = high_resolution_clock::now();

    sem_init(&configSem, 0, 1);

    vector<DataThread*> dataThreads;
    map<int, map<string, stats>> statsMap = initializeStats();

    auto inpStart = high_resolution_clock::now();

    for (int i = 0; i < NUM_OF_THREADS; i++) {
        DataThread* dt = new DataThread;
        dt->threadNumber = i;
        dt->dataThread = thread(getParallelInput, dt, directory);
        dataThreads.push_back(dt);
    }

    joinThreads(dataThreads);

    map<int, int> numRows;
    numRows[0] = 0;
    numRows[1] = 0;

    mergeMeanVals(dataThreads, statsMap, numRows);

    for (int i = 0; i < NUM_OF_THREADS; i++) {
        dataThreads[i]->dataThread
            = thread(getThreadStdSum, dataThreads[i]);
    }

    joinThreads(dataThreads);

    mergeStdVals(dataThreads, statsMap, numRows);

    auto inpEnd = high_resolution_clock::now();
    auto outStart = high_resolution_clock::now();

    for (int i = 0; i < NUM_OF_THREADS; i++) {
        dataThreads[i]->dataThread
            = thread(getParallelOutput, dataThreads[i], statsMap);
    }

    joinThreads(dataThreads);

    double result = calculateAccuracy(dataThreads);
    auto outEnd = high_resolution_clock::now();
    auto totalEnd = high_resolution_clock::now();

    printf(PRINT_FORMAT, result*100);

    auto totalDuration = duration_cast<microseconds>(totalEnd - totalStart);
    auto inpDuration = duration_cast<microseconds>(inpEnd - inpStart);
    auto outDuration = duration_cast<microseconds>(outEnd - outStart);

    return result;
}
