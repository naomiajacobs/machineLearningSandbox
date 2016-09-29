import pdb
import numpy as np
import csv
import argparse

parser = argparse.ArgumentParser(description='Finds a function to predict future values based on training data.')
parser.add_argument('csvFile', metavar='csv', type=argparse.FileType('r'), help='path to csv file')

args = parser.parse_args()
csvFile = args.csvFile

dataPoints = []
actualResults = []

with csvFile as file:
    rows = csv.reader(file, dialect='excel')
    for i, row in enumerate(rows):
        if i != 0:
            numbers = map(float, row)
            dataPoints.append(numbers[:-1])
            actualResults.append(numbers[-1])

num_of_features = len(dataPoints[0])
intercept = num_of_features
currentTheta = list(range(num_of_features + 1))
hypothesizedResults = list(range(len(dataPoints)))
errorTerms = list(range(len(dataPoints)))

# initializeTheta
for i in range(num_of_features + 1): # one extra for intercept
    currentTheta[i] = 0.5

trainingRate = -1

def recalculateTheta():
    for featureIndex, coefficient in enumerate(currentTheta):
        sumTerm = 0
        for dataIndex, dataPoint in enumerate(dataPoints):
            featureDatum = dataPoint[featureIndex] if featureIndex != (len(currentTheta) - 1) else 1
            sumTerm += errorTerms[dataIndex] * featureDatum
        newCoefficient = coefficient + (trainingRate * sumTerm)
        currentTheta[featureIndex] = newCoefficient

def runIteration():
    for index, dataPoint in enumerate(dataPoints):
        result = 0
        for i in range(num_of_features):
            result += dataPoint[i] * currentTheta[i]
        result += currentTheta[intercept] # constant theta vector
        hypothesizedResults[index] = result

        # calculate error terms
        errorTerms[index] = actualResults[index] - hypothesizedResults[index]

    # print('Error terms: ', errorTerms)
    print('Current theta: ', currentTheta)

for i in range(100):
    runIteration()
    recalculateTheta()
