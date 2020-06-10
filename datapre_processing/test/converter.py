__author__ = 'Marco De Nadai'
__license__ = "MIT"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import datetime
import csv
from collections import namedtuple
from collections import defaultdict
import fiona
from shapely.geometry import shape, Polygon


# Import the CDRs of MILANO
df = pd.read_csv('datasets/MILANO_CDRs.csv', sep=',', encoding="utf-8-sig", parse_dates=['datetime'])

# Considered technologies list
TECHNOLOGIES = ['GSM1800']

# Import the technologies' coverage areas
# Note: the input files cannot be shared.
coverage_polygons = []
for t in TECHNOLOGIES:
    source = fiona.open('datasets/shapefiles/COVERAGE_'+t+'.shp', 'r')
    for polygon in source:
        coverage_polygons.append(shape(polygon['geometry']))

# Create squares of 1 km^2
# ref: http://stackoverflow.com/questions/4000886/gps-coordinates-1km-square-around-a-point
earth_circumference = math.cos(math.radians(df['lat'].mean()))*40075.16
# Let's create squares of 235x235 metres
kms = 0.235
gridWidth = math.float(kms * (360./earth_circumference))
gridHeight = math.float(kms/111.32)

# GRID bounds (coordinates)
XMIN = 9.011533669936474
YMIN = 45.356261753717845
XMAX = 9.312688264185276
YMAX = 45.56821407553667

# Get the number of rows and columns
rows = math.ceil((YMAX-YMIN)/gridHeight)
cols = math.ceil((XMAX-XMIN)/gridWidth)


Square = namedtuple('Square', ['x', 'y', 'cx', 'cy', 'polygon'])
square_grid = []
for i in range(int(rows)):
    for j in range(int(cols)):
        x = XMIN+j*gridWidth
        y = YMIN+i*gridHeight
        centerx = (x+x+gridWidth)/2.
        centery = (y+y+gridHeight)/2.

        p = Polygon([[x,y], [x, y+gridHeight], [x+gridWidth, y+gridHeight], [x+gridWidth, y]])

        square_grid.append(Square(x, y, centerx, centery, p))

# Calculate the intersections of the coverage cells with the grids' square
intersections = []
for t in TECHNOLOGIES:
    for i, v in enumerate(coverage_polygons[t]):
        total_coverage_area = v.polygon.area
        for j, s in enumerate(square_grid):
            if v.polygon.intersects(s.polygon):

                # To avoid Python floating point errors
                if s.polygon.contains(v.polygon):
                    fraction = 1.0
                else:
                    # Calculates the proportion between the intersection between the coverage and the grid
                    # square. This is useful to assign the right proportion of the the mobile usage to the
                    # grid square.
                    fraction = (v.polygon.intersection(s.polygon).area/total_coverage_area)

                coverage_polygons[t][i].intersections.append([j, fraction])

coverage_intersections = defaultdict(dict)
for t in TECHNOLOGIES:
    coverage_intersections[t] = defaultdict(dict)
    for p in coverage_polygons[t]:
        coverage_intersections[t][p.CGI] = p.intersections


# We build a hash table to search in a fast way all the CGI of a technology
hash_cgi_tech = {}
for index,row in df.groupby(['cgi','technology'], as_index=False).sum().iterrows():
    hash_cgi_tech[row['cgi']] = row['technology']


# Select the data grouped by hour and countrycode
groups = df.groupby(['datetime', 'countrycode'])

#
# Example file with the format:
# datetime,CGI,countryCode,numRecords
#
with open('dati/MILANO_grid.csv', 'wb') as csvfile:
    csvwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    csvwriter.writerow(["datetime", "GridCell", "countryCode", "numRecords"])

    for name, group in groups:
        # iterate group's rows
        data = []
        d = defaultdict(int)
        for index, row in enumerate(group.values):
            CGI = row[1]
            tech = hash_cgi_tech[CGI]

            if CGI in coverage_intersections[tech]:
                for (cell_number, cell_intersection_portion) in coverage_intersections[tech][CGI]:
                    d[str(cell_number) + "_" + str(row[3])] += float(row[2]*cell_intersection_portion)

        datetime_rows = group.values[0, 0]
        rows = [[datetime_rows] + k.split("_") + [v] for (k, v) in d.iteritems()]

        csvwriter.writerows(rows)