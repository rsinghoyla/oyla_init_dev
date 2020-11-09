from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import pyqtSignal, Qt, QThread, pyqtSlot
import sys
def main():
    global Thread
    app = QtWidgets.QApplication(sys.argv)
    main = Main()
    Thread = RFID_Thread()
    Thread.rfid_event.connect(Main().on_event)
    Thread.start()
    sys.exit(app.exec_())

class Main(object):
    def __init__(self):
        self.accueil = MainWindow(self)
        self.access = AccessWindow()
        self.accueil.show()

    def on_event(self, data):
        # I WANT TO PAUSE THE QTHREAD HERE

        Thread.Pause = False
        ###################################
        #   CHECKING DB & SHOWING UI      #
        ###################################

        # AND RESUME IT HERE
        Thread.Pause = True
class RFID_Thread(QtCore.QThread):
    rfid_event = pyqtSignal(str, name='rfid_event')
    Pause = True
    def run(self):
        while 1:
            if Pause:
                  ser = serial.Serial(port=Serial_Port, baudrate=Serial_Baudrate)
                  a = ser.read(19).encode('hex')
                  ser.close()
                  if len(a) != 0:
                         Code = a[14:]
                         self.rfid_event.emit(Code)
                         time.sleep(2)
            else:
                  continue

if __name__=='__main__':
    main()
