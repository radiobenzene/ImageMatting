import sys
from PyQt5.QtWidgets import QApplication, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget

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

    sys.exit(app.exec_())
