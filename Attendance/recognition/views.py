
import csv
import re
from unittest import result
import cv2
from django.contrib.auth.decorators import login_required
import pandas as pd


# # Create your views here.
from django.shortcuts import redirect, render
from django.template import Context

from recognition.forms import usernameForm


def home(request):

	return render(request, 'home.html')

@login_required
def dashboard(request):
	if(request.user.username=='admin'):
		print("admin")
		return render(request, 'admin_dashboard.html')
	else:
		print("not admin")

		return render(request,'employee_dashboard.html')



import face_recognition
import os
from datetime import datetime
from datetime import date
import numpy as np

def findEncodings(images):
		encodeList=[]

		for img in images:
			img2 = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

			if len(face_recognition.face_encodings(img2))!=0:
				encode = face_recognition.face_encodings(img2)[0]
				encodeList.append(encode)
			print(encodeList)
		return encodeList



def markAttendance(name):
		with open('recognition\AttendanceIn.csv','r+') as f:
			myDataList = f.readlines()
			nameList = []
			for line in myDataList:
				entry = line.split(',')
				nameList.append(entry[0])
				print(nameList)
            
			now = datetime.now()
			dt = date.today()
			dateString= dt.strftime('%d/%m/%Y')
			dtString = now.strftime('%H:%M:%S')
			print(dateString)
			f.writelines(f'\n{name},{dateString},{dtString}')

def markAttendanceOut(name):
		with open('recognition\AttendanceOut.csv','r+') as f:
			myDataList = f.readlines()
			nameList = []
			for line in myDataList:
				entry = line.split(',')
				nameList.append(entry[0])
            
			now = datetime.now()
			dt = date.today()
			dateString= dt.strftime('%d/%m/%Y')
			dtString = now.strftime('%H:%M:%S')
			print(dateString)
			f.writelines(f'\n{name},{dateString},{dtString}')

def mark_your_attendance(request):
	flagin=1
	count=0
	path = 'recognition\ImagesAttendance'
	images = []
	classNames = []
	myList = os.listdir(path)
	print(myList)

    
	for cl in myList:
		curImg = cv2.imread(f'{path}/{cl}')
		images.append(curImg)
		classNames.append(os.path.splitext(cl)[0])

	print(classNames)

	encodeListKnown = findEncodings(images)
	print("Encoding complete")

	cap = cv2.VideoCapture(0)

	while flagin==1 and count<3:
		count+=1
		success,img = cap.read()
		imgS = cv2.resize(img,(0,0),None,0.25,0.25)
		imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

		facesCurFrame = face_recognition.face_locations(imgS)
		encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)

		for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
			matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
			faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
			#print(faceDis)
			matchIndex = np.argmin(faceDis)

			if matches[matchIndex]:
				
				name = classNames[matchIndex].upper()
				#print(name)
				y1,x2,y2,x1 = faceLoc
				y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
				cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
				cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
				cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
				cv2.putText(img,'Successful!',(200,450),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
				# cv2.waitKey(1)
				
				if flagin==1:
					markAttendance(name)
					flagin=0

			else: 
				y1,x2,y2,x1 = faceLoc
				y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
				cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),2)
				cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,0,255),cv2.FILLED)
				cv2.putText(img,'FAIL',(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
				cv2.putText(img,'Attempt:'+str(count),(20,40),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),2)

				cv2.putText(img,'Unregisterd User',(200,450),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
				cv2.waitKey(5)
				if count>=3 and flagin==1:
					flagin=0
				

			

		cv2.imshow('Webcam',img)
		cv2.waitKey(2500)  
		

	return redirect('home')





def mark_your_attendance_out(request):
	flagout=1
	count=1

	
	path = 'recognition\ImagesAttendance'
	images = []
	classNames = []
	myList = os.listdir(path)
	print(myList)


	for cl in myList:
		curImg = cv2.imread(f'{path}/{cl}')
		images.append(curImg)
		classNames.append(os.path.splitext(cl)[0])

	print(classNames)

	encodeListKnown = findEncodings(images)
	print("Encoding complete")

	cap = cv2.VideoCapture(0)

	while flagout==1 and count<3:
		count+=1
		success,img = cap.read()
		imgS = cv2.resize(img,(0,0),None,0.25,0.25)
		imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

		facesCurFrame = face_recognition.face_locations(imgS)
		encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)

		for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
			matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
			faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
			#print(faceDis)
			matchIndex = np.argmin(faceDis)

			if matches[matchIndex]:
				
				name = classNames[matchIndex].upper()
				#print(name)
				y1,x2,y2,x1 = faceLoc
				y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
				cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
				cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
				cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
				cv2.putText(img,'Successful!',(200,450),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
				# cv2.waitKey(1)
				
				if flagout==1:
					markAttendanceOut(name)
					flagout=0

			else: 
				y1,x2,y2,x1 = faceLoc
				y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
				cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),2)
				cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,0,255),cv2.FILLED)
				cv2.putText(img,'FAIL',(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
				cv2.putText(img,'Attempt:'+str(count),(20,40),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),2)

				cv2.putText(img,'Unregisterd User',(200,450),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
				cv2.waitKey(5)
				if count>=3 and flagout==1:
					flagout=0
				

			

		cv2.imshow('Webcam',img)
		cv2.waitKey(2500)
		

	return redirect('home')

    
# @login_required
# def not_authorised(request):
# 	return render(request,'not_authorised.html')



# @login_required
def view_attendance_in(request):
	csv_fp = pd.read_csv('recognition\AttendanceIn.csv',header=0)
	context2 = csv_fp.to_dict('list')
	result_set = dict()
	result_set['data'] = context2
	result_set['header'] =  [v for v in context2.keys()]
	return render(request, 'view_attendance_in.html', result_set)
	
	
def view_attendance_home(request):

	csv_fp = pd.read_csv('recognition\AttendanceOut.csv',header=0)
	context2 = csv_fp.to_dict('list')
	result_set = dict()
	result_set['data'] = context2
	result_set['header'] =  [v for v in context2.keys()]
	return render(request, 'view_attendance_home.html', result_set)
	
    
	