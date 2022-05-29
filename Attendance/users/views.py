from tkinter import image_types
from urllib import request
from django.shortcuts import render,redirect
from django.contrib.auth.forms import UserCreationForm
from .forms import CustomUserCreationForm  
from django.contrib import messages
from django.contrib.auth.decorators import login_required


#utility functions
'''
def hours_vs_date_every_employee():
	qs = Attendance.objects.all()
	diff=[]
	
	for obj in qs:
		ti=obj.time_in
		to=obj.time_out
		hours=((to-ti).total_seconds())/3600
		diff.append(hours)
		
	df = read_frame(qs)
	df['hours']=diff
	figure=plt.figure()
	sns.barplot(data=df,x='date',y='')
	html_graph=mpld3.fig_to_html(fig)


'''


# Create your views here.



def register(request): 
	context={}
	if request.POST == 'POST':  
		form = CustomUserCreationForm()  
		if form.is_valid():  
			form.save()  
	else:  
		form = CustomUserCreationForm()  
	context = {  
		'form':form  } 


	print(context)
	return render(request, 'register.html', context)  




import cv2 
def clickPicture(request):
	key = cv2. waitKey(1)
	webcam = cv2.VideoCapture(0)
	user_name = request.POST['user_name']
	while True:
		try:
			check, frame = webcam.read()
			# print(check) #prints true as long as the webcam is running
			# print(frame) #prints matrix values of each framecd 
			cv2.imshow("Capturing", frame)
			key = cv2.waitKey(1)
			if key == ord('s'): 
				image_path = 'recognition/ImagesAttendance/'+user_name+'.jpg'
				cv2.imwrite(filename=image_path, img=frame)
				webcam.release()
			
				img_new = cv2.imread(image_path)
				img_new = cv2.imshow("Captured Image", img_new)
				cv2.waitKey(1650)
				cv2.destroyAllWindows()
				break
			elif key == ord('q'):
				print("Turning off camera.")
				webcam.release()
				print("Camera off.")
				print("Program ended.")
				cv2.destroyAllWindows()
				break
			
		except(KeyboardInterrupt):
			print("Turning off camera.")
			webcam.release()
			print("Camera off.")
			print("Program ended.")
			cv2.destroyAllWindows()
			break
		
	return render(request,'register.html')


