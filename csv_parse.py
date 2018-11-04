import csv
import sys
csv.field_size_limit(sys.maxsize)
with open('8903572_csv_2018_11_03_ed770.csv') as csvfile:
    linereader = csv.reader(csvfile, delimiter=',', quotechar='"')
    for row in linereader:
        print("url : %s ** tag : %s " % (row[1], row[2]))


