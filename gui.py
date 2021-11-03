from PyQt5.QtWidgets import (QWidget, QGridLayout, QVBoxLayout, QPushButton,
    QDesktopWidget, QGroupBox, QStyleFactory, QApplication)
from PyQt5.QtGui import QFont
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import networkx as nx

class PrettyWidget(QWidget):

    def __init__(self):
        super().__init__()     
        font = QFont()
        font.setPointSize(16)
        self.init_ui()

    def init_ui(self):
        self.setGeometry(100, 100, 800, 600)
        self.center()
        self.setWindowTitle('MARLon')

        grid = QGridLayout()
        self.setLayout(grid)
        self.create_vertical_box()

        button_layout = QVBoxLayout()
        button_layout.addWidget(self.vertical_box)

        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)  
        grid.addWidget(self.canvas, 0, 1, 9, 9)    
        grid.addLayout(button_layout, 0, 0)

        self.plot()
        self.show()


    def create_vertical_box(self):
        self.vertical_box = QGroupBox()

        layout = QVBoxLayout()

        next_button = QPushButton('Next')
        layout.addWidget(next_button)

        prev_button = QPushButton('Previous')
        layout.addWidget(prev_button)

        layout.setSpacing(10)
        self.vertical_box.setLayout(layout)

    def plot(self):
        self.figure.clf()
        b = nx.Graph()
        b.add_nodes_from([1, 2, 3, 4], bipartite=0)
        b.add_nodes_from(['a', 'b', 'c', 'd', 'e'], bipartite=1)
        b.add_edges_from([(1, 'a'), (2, 'c'), (3, 'd'), (3, 'e'), (4, 'e'), (4, 'd')])

        x = set(n for n, d in b.nodes(data=True) if d['bipartite'] == 0)
        y = set(b) - x

        x = sorted(x, reverse=True)
        y = sorted(y, reverse=True)

        pos = {}
        pos.update( (n, (1, i)) for i, n in enumerate(x) ) # put nodes from X at x=1
        pos.update( (n, (2, i)) for i, n in enumerate(y) ) # put nodes from Y at x=2
        nx.draw(b, pos=pos, with_labels=True)
        self.canvas.draw_idle()

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

if __name__ == '__main__':

    import sys
    app = QApplication(sys.argv)
    app.aboutToQuit.connect(app.deleteLater)
    app.setStyle(QStyleFactory.create("gtk"))
    screen = PrettyWidget()
    screen.show()
    sys.exit(app.exec_())
