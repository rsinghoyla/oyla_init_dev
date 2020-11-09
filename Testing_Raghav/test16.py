from sys import argv

from PyQt5.QtCore import QObject, pyqtSignal, QThread, Qt, QMutex

from PyQt5.QtWidgets import QDialog, QApplication, QPushButton, \
     QLineEdit, QFormLayout, QTextEdit

class TestTask(QObject):

    def __init__(self, parent=None):
        QObject.__init__(self, parent)

        self._mutex = QMutex()

        self._end_loop = True

    def init_object(self):
        while self._end_loop:
            QThread.msleep(300)
            QApplication.processEvents()
            print("Sratus", self._end_loop)

    def stop(self):
        self._mutex.lock()
        self._end_loop = False
        self._mutex.unlock()

class Form(QDialog):
    stop_loop = pyqtSignal()

    def __init__(self, parent=None):
        QDialog.__init__(self, parent)

        self.init_ui()

    def init_ui(self):


        self.pushButton_start_loop = QPushButton()
        self.pushButton_start_loop.setText("Start Loop") 

        self.pushButton_stop_loop = QPushButton()
        self.pushButton_stop_loop.setText("Stop Loop")       

        self.pushButton_close = QPushButton()
        self.pushButton_close.setText("Close")

        layout = QFormLayout()

        layout.addWidget(self.pushButton_start_loop)
        layout.addWidget(self.pushButton_stop_loop)
        layout.addWidget(self.pushButton_close)

        self.setLayout(layout)
        self.setWindowTitle("Tes Window")

        self.init_signal_slot_pushButton()

    def start_task(self):

         self.task_thread = QThread(self)
         self.task_thread.work = TestTask()
         print(self.task_thread)
         self.task_thread.work.moveToThread(self.task_thread)
         self.task_thread.started.connect(self.task_thread.work.init_object)

         self.stop_loop.connect(self.task_thread.work.stop)

         self.task_thread.start()

    def stop_looping(self):
        self.stop_loop.emit()

    def init_signal_slot_pushButton(self):

        self.pushButton_start_loop.clicked.connect(self.start_task)

        self.pushButton_stop_loop.clicked.connect(self.stop_looping)

        self.pushButton_close.clicked.connect(self.close)



app = QApplication(argv)
form = Form()
form.show()
app.exec_()
