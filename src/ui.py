import sys
from PyQt5.QtWidgets import *
from PyQt5 import QtCore, QtGui, QtWidgets
import crawler


class MyWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setupUI()
    def setupUI(self):
        self.setGeometry(800, 200, 300, 300)
        self.instavalue = 0
        self.facevalue = 0
        self.pintervalue = 0
        
#        
        font = QtGui.QFont()
        font.setFamily("맑은 고딕")
        font.setBold(True)
        font.setWeight(75)
        self.setFont(font)
        self.setWindowTitle("키워드에 따른 브랜드 통계")
#


#바꾼코드 -(1)
        self.setMinimumSize(QtCore.QSize(400, 600))
        self.setMaximumSize(QtCore.QSize(400, 600))
        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setObjectName("centralwidget")
#바꾼코드 -       
        self.lineEdit = QLineEdit("", self)
        self.lineEdit.move(10, 20)
        self.lineEdit.textChanged.connect(self.lineEditChanged)
#바꾼코드 -


        
#바꾼코드 -(2)
        btn1 = QPushButton("검색", self)
        btn1.move(280, 20)
        btn1.clicked.connect(self.actionstart)
#바꾼코드 -(2)        
        
#바꾼코드 -(4)
        self.checkBox1 = QCheckBox("샤넬", self)
        self.checkBox1.move(10, 90)
        self.checkBox1.resize(150, 30)
        self.checkBox1.stateChanged.connect(self.checkBoxState)

        self.checkBox2 = QCheckBox("닥터자르트", self)
        self.checkBox2.move(10, 120)
        self.checkBox2.resize(150, 30)
        self.checkBox2.stateChanged.connect(self.checkBoxState)

        self.checkBox3 = QCheckBox("이니스프리", self)
        self.checkBox3.move(10, 150)
        self.checkBox3.resize(150, 30)
        self.checkBox3.stateChanged.connect(self.checkBoxState)

        self.checkBox4 = QCheckBox("헤라", self)
        self.checkBox4.move(10, 180)
        self.checkBox4.resize(150, 30)
        self.checkBox4.stateChanged.connect(self.checkBoxState)

        self.checkBox5 = QCheckBox("라네즈", self)
        self.checkBox5.move(10, 210)
        self.checkBox5.resize(150, 30)
        self.checkBox5.stateChanged.connect(self.checkBoxState)

        self.checkBox6 = QCheckBox("나스", self)
        self.checkBox6.move(10, 240)
        self.checkBox6.resize(150, 30)
        self.checkBox6.stateChanged.connect(self.checkBoxState)

        self.checkBox7 = QCheckBox("설화수", self)
        self.checkBox7.move(10, 270)
        self.checkBox7.resize(150, 30)
        self.checkBox7.stateChanged.connect(self.checkBoxState)

        self.checkBox8 = QCheckBox("숨", self)
        self.checkBox8.move(10, 300)
        self.checkBox8.resize(150, 30)
        self.checkBox8.stateChanged.connect(self.checkBoxState)

        self.checkBox9 = QCheckBox("후", self)
        self.checkBox9.move(10, 330)
        self.checkBox9.resize(150, 30)
        self.checkBox9.stateChanged.connect(self.checkBoxState)

        self.checkBox10 = QCheckBox("입생로랑", self)
        self.checkBox10.move(10, 360)
        self.checkBox10.resize(150, 30)
        self.checkBox10.stateChanged.connect(self.checkBoxState)
#바꾼코드 -(4)
    
#원코드
        self.checkBox11 = QCheckBox("인스타그램", self)
        self.checkBox11.move(200, 90)
        self.checkBox11.resize(150, 30)
        self.checkBox11.stateChanged.connect(self.checkBoxState)

        self.checkBox12 = QCheckBox("페이스북", self)
        self.checkBox12.move(200, 120)
        self.checkBox12.resize(150, 30)
        self.checkBox12.stateChanged.connect(self.checkBoxState)

        self.checkBox13 = QCheckBox("핀터레스트", self)
        self.checkBox13.move(200, 150)
        self.checkBox13.resize(150, 30)
        self.checkBox13.stateChanged.connect(self.checkBoxState)

        self.statusBar = QStatusBar(self)
        self.setStatusBar(self.statusBar)
#

#
    def checkBoxState(self):
        msg = ""

#        
        if self.checkBox11.isChecked() == True:
            msg += "<인스타>"
            self.instavalue = 1
        else :
            self.instavalue = 0
#

#
        if self.checkBox12.isChecked() == True:
            msg += "<페이스북>"
            self.facevalue = 1
        else :
            self.facevalue = 0
#

#
        if self.checkBox13.isChecked() == True:
            msg += "<핀터레스트>"
            self.pintervalue = 1
        else :
            self.pintervalue = 0
#            
        self.statusBar.showMessage(msg)
#

#
    def lineEditChanged(self):
        self.statusBar.showMessage(self.lineEdit.text())
#

#
    def actionstart(self) : 
        if self.instavalue == 1 :
            crawler.crawl(self.lineEdit.text(), 5, 0)
#

#
def run():
    app = QApplication(sys.argv)
    mywindow = MyWindow()
    mywindow.show()
    app.exec_()
