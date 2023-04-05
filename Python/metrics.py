import sys
from PyQt5.QtWidgets import QApplication, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget
import csv
from PyQt5.QtWidgets import QApplication, QMainWindow, QSizePolicy
from PyQt5.QtGui import QPainter
from PyQt5.QtCore import Qt, QDateTime
import matplotlib.pyplot as plt
from datetime import datetime

'''
    Function to display SAD 
'''
def displaySAD(file_path):
    with open(file_path, 'r') as f:
        data = [line.strip().split(',') for line in f]

    app = QApplication(sys.argv)
    table = QTableWidget()
    table.setRowCount(len(data))
    table.setColumnCount(2)
    table.setHorizontalHeaderLabels(['Image Name', 'SAD Values'])

    for i, row in enumerate(data):
        table.setItem(i, 0, QTableWidgetItem(row[0]))
        table.setItem(i, 1, QTableWidgetItem(row[1]))

    layout = QVBoxLayout()
    layout.addWidget(table)

    widget = QWidget()
    widget.setLayout(layout)
    widget.show()
    
    generateGraph(file_path, 'SAD', 'SAD')


    sys.exit(app.exec_())
    
'''
    Function to display Execution Time
'''
def displayExecutionTime(file_path):
    with open(file_path, 'r') as f:
        data = [line.strip().split(',') for line in f]

    app = QApplication(sys.argv)
    table = QTableWidget()
    table.setRowCount(len(data))
    table.setColumnCount(2)
    table.setHorizontalHeaderLabels(['Image Name', 'Execution Time '])

    for i, row in enumerate(data):
        table.setItem(i, 0, QTableWidgetItem(row[0]))
        table.setItem(i, 1, QTableWidgetItem(row[1]))

    layout = QVBoxLayout()
    layout.addWidget(table)

    widget = QWidget()
    widget.setLayout(layout)
    widget.show()
    
    generateGraph(file_path, 'Execution Time', 'Execution Time')

    sys.exit(app.exec_())
'''
    Function to display metric
'''
def displayMetric(file_path, title, xlabel, ylabel):
    with open(file_path, 'r') as f:
        data = [line.strip().split(',') for line in f]

    app = QApplication(sys.argv)
    table = QTableWidget()
    table.setRowCount(len(data))
    table.setColumnCount(2)
    #table.setHorizontalHeaderLabels(['Image Name', 'Execution Time '])
    table.setHorizontalHeaderLabels([xlabel, ylabel])
    for i, row in enumerate(data):
        table.setItem(i, 0, QTableWidgetItem(row[0]))
        table.setItem(i, 1, QTableWidgetItem(row[1]))

    layout = QVBoxLayout()
    layout.addWidget(table)

    widget = QWidget()
    widget.setLayout(layout)
    widget.show()
    
    generateGraph(file_path, title, ylabel)

    sys.exit(app.exec_())
    
'''
    Function to generate graph
'''
def generateGraph(data_file, title, ylabel_val):
    # Read data from the file
    data = {}
    with open(data_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            image_name, mse = row
            data[image_name] = float(mse)
    
    # Create lists of x and y values
    x_values = []
    y_values = []
    for image_name, mse in data.items():
        x_values.append(image_name)
        y_values.append(mse)
    
    # Create line graph using Matplotlib
    plt.plot(x_values, y_values)
    plt.title(title)
    plt.xlabel('Image name')
    plt.ylabel(ylabel_val)
    plt.xticks(rotation=45)
    
    plt.show()
