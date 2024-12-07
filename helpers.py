import csv

def toCSV(l1, l2, l3, l4):
    with open("output.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows([l1, l2, l3, l4])