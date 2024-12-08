import csv
import json

def myelinToggles():
    with open('ExternalData/myelinToggles.json', 'r') as file:
        data = json.load(file)
        return {
            "isSensoryMyelinated": data['SensoryMyelination'],
            "isExtensorMyelinated": data['ExtensorMyelination'],
            "isInhibitoryMyelinated": data['InhibitoryMyelination'],
            "isFlexorMyelinated": data['FlexorMyelination'],
        }

def toCSV(l1, l2, l3, l4):
    with open("ExternalData/output.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows([l1, l2, l3, l4])