#include "dataProcessor.hpp"

using namespace std::chrono;
using namespace std;

struct DatasetConfig {
    vector<string> FIELD_NAMES;
    int NUM_OF_FIELDS;
    bool isSet = false;
    int NUM_OF_ROWS;
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
    DATASET_CONFIG.NUM_OF_ROWS = size;
}

void setDatasetTargetName(string targetName) {
    DATASET_CONFIG.TARGET_NAME = targetName;
}

void setDatasetTargetThreshold(double threshold) {
    DATASET_CONFIG.TARGET_THRESHOLD = threshold;
}

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
    if (!DATASET_CONFIG.isSet) {
        setDatasetConfig(firstLine);
    }
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
        row.insert(
            pair<string, double>(DATASET_CONFIG.FIELD_NAMES[fieldCounter],
            stod(word))
        );
        dataValues.push_back(row);
	}
	fin.close();
    setDatasetSize(numRows);
    setDatasetTargetName(DATASET_CONFIG.FIELD_NAMES[DATASET_CONFIG.NUM_OF_FIELDS - 1]);
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

void getStats(
    vector<map<string, double>>& dataset,
    map<int, map<string, stats>>& statsMap
) {
    map<int, int> numRows;
    numRows[0] = 0;
    numRows[1] = 0;
    vector<map<string, double>>::iterator vecItr;

    for (vecItr = dataset.begin(); vecItr != dataset.end(); ++vecItr) {
        updateMeanSum(*vecItr, statsMap, numRows);
    }
    updateMean(statsMap, numRows);
     for (vecItr = dataset.begin(); vecItr != dataset.end(); ++vecItr) {
        updateStdSum(*vecItr, statsMap);
    }
    updateStd(statsMap, numRows);
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

double calculateAccuracy(vector<double>& predictions, vector<map<string, double>>& dataset) {
    int count = 0;
    vector<map<string, double>>::iterator vecItr;
    int i = 0;
    for (vecItr = dataset.begin(); vecItr != dataset.end(); ++vecItr) {
        if ((*vecItr)[DATASET_CONFIG.TARGET_NAME] == predictions[i])
            count++;
        i++;
    }
    return (double) count / i;
}

double runDataProcessing(string directory, double threshold) {
    auto totalStart = high_resolution_clock::now();

    vector<map<string, double>> dataset;
    vector<double> foundTargets;
    double result;

    setDatasetTargetThreshold(threshold);

    auto inpStart = high_resolution_clock::now();
    readCSV(directory + DATASET_PATH, dataset);
    auto inpEnd = high_resolution_clock::now();

    auto labelStart = high_resolution_clock::now();
    labelizeTarget(dataset);
    auto labelEnd = high_resolution_clock::now();

    auto statsStart = high_resolution_clock::now();
	map<int, map<string, stats>> statsMap = initializeStats();
    getStats(dataset, statsMap);
    auto statsEnd = high_resolution_clock::now();


    auto targetStart = high_resolution_clock::now();
    predictAllTargets(foundTargets, dataset, statsMap);
    auto targetEnd = high_resolution_clock::now();

    auto accuracyStart = high_resolution_clock::now();
    result = calculateAccuracy(foundTargets, dataset);
    auto accuracyEnd = high_resolution_clock::now();

    printf(PRINT_FORMAT, result*100);

    auto totalEnd = high_resolution_clock::now();

    auto totalDuration = duration_cast<microseconds>(totalEnd - totalStart);
    auto inpDuration = duration_cast<microseconds>(inpEnd - inpStart);
    auto statsDuration = duration_cast<microseconds>(statsEnd - statsStart);
    auto labelDuration = duration_cast<microseconds>(labelEnd - labelStart);
    auto targetDuration = duration_cast<microseconds>(targetEnd - targetStart);
    auto accuracyDuration = duration_cast<microseconds>(accuracyEnd - accuracyStart);

    // cout << "Total Time " << totalDuration.count() << " microseconds" << endl;
    // cout << "Inp Time " << inpDuration.count() << " microseconds" << endl;
    // cout << "stats Time " << statsDuration.count() << " microseconds" << endl;
    // cout << "label Time " << labelDuration.count() << " microseconds" << endl;
    // cout << "Target Time " << targetDuration.count() << " microseconds" << endl;
    // cout << "Accuracy Time " << accuracyDuration.count() << " microseconds" << endl;

    return result;
}
