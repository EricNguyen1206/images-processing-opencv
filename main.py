import cv2
import imutils
import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk
import PIL
import cv2

from contrast_stretching import contrast_stretching
from face_detection import face_detect
from face_recognition import face_recognition

def get_image_from_dialog():
  pass

#initialise GUI
top=tk.Tk()
top.geometry('800x600')
top.title('Face recognition')
top.configure(background='#CDCDCD')

label=Label(top,background='#CDCDCD', font=('Roboto',15,'bold'))
sign_image = Label(top)

def show_classify_button(file_path):
    classify_b = Button(top,text="Recognition faces",command=lambda: classify(file_path),padx=10,pady=5)
    classify_b.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
    classify_b.place(relx=0.79,rely=0.46)

def upload_image():
    try:
        file_path=filedialog.askopenfilename()
        uploaded=PIL.Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.25),(top.winfo_height()/2.25)))
        im=ImageTk.PhotoImage(uploaded)

        sign_image.configure(image=im)
        sign_image.image=im
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass

def classify(file_path):
  # read and resize image
  img = cv2.imread(file_path)
  img = imutils.resize(img, width=500)

  # contrast stretching and de-noise
  new_img = contrast_stretching(img)
  cv2.imshow('new_img', new_img)

  # edge detection

  # face detection
  coordinator_list, img = face_detect(img)

  # face recognition
  face_recognition(coordinator_list, img)
  # label.configure(foreground='#011638', text=sign) 
  
  sign_image.configure(image=img)
  sign_image.image=img
  #


if __name__ == '__main__':
  upload=Button(top,text="Upload an image",command=upload_image,padx=10,pady=5)
  upload.configure(background='#364156', foreground='white',font=('arial',10,'bold'))

  upload.pack(side=BOTTOM,pady=50)
  sign_image.pack(side=BOTTOM,expand=True)
  label.pack(side=BOTTOM,expand=True)
  heading = Label(top, text="Face_recognition",pady=20, font=('arial',20,'bold'))
  heading.configure(background='#CDCDCD',foreground='#364156')
  heading.pack()
  top.mainloop()


