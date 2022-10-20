import cv2
from cv2 import denoise_TVL1
import imutils
import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk
import PIL
import cv2

from noise_reduce import noise_reduce
from data_processing import adjust_gamma, sharping
from contrast_stretching import contrast_stretching
from face_detection import face_detect
from face_recognition import face_recognition
from edge_detection import edge_detect

#initialise GUI
top=tk.Tk()
top.geometry('800x600')
top.title('Face recognition')
top.configure(background='#CDCDCD')

label=Label(top,background='#CDCDCD', font=('Roboto',15,'bold'))
sign_image = Label(top)

def show_classify_button(file_path):
    rel_x_root = 0.79
    rel_y_root = 0.26

    classify_b_none = Button(top,text="Normal",command=lambda: classify(file_path, None),padx=10,pady=5)
    classify_b_bilateral = Button(top,text="Bilateral", command=lambda: classify(file_path, 'bilateral'),padx=10,pady=5)
    classify_b_median = Button(top,text="Median",command=lambda: classify(file_path, 's&p'),padx=10,pady=5)
    classify_b_mean = Button(top,text="Mean",command=lambda: classify(file_path, 'mean'),padx=10,pady=5)
    classify_b_gauss = Button(top,text="Gaussian",command=lambda: classify(file_path, 'gauss'),padx=10,pady=5)

    classify_b_none.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
    classify_b_none.place(relx=rel_x_root,rely=rel_y_root)

    classify_b_bilateral.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
    classify_b_bilateral.place(relx=rel_x_root, rely=rel_y_root+0.1)

    classify_b_median.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
    classify_b_median.place(relx=rel_x_root, rely=rel_y_root+0.2)

    classify_b_gauss.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
    classify_b_gauss.place(relx=rel_x_root, rely=rel_y_root+0.3)

    classify_b_mean.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
    classify_b_mean.place(relx=rel_x_root, rely=rel_y_root+0.4)

def upload_image():
    try:
        file_path=filedialog.askopenfilename()
        uploaded=PIL.Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.25),(top.winfo_height()/2.25)))
        im=ImageTk.PhotoImage(uploaded)

        sign_image.configure(image=im)
        sign_image.image=im
        label.configure(text='Origin image')

        show_classify_button(file_path)
    except:
        pass

def classify(file_path, noise_type):
  # read and resize image
  img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
  img = imutils.resize(img, height=500)


  # noise reduction
  denoise_img = noise_reduce(noise_type, img)

  # adjust brightness
  denoise_img = adjust_gamma(denoise_img, 1)

  # contrast stretching and de-noise
  contrast_stretching_img = contrast_stretching(denoise_img)

  # sharping 
  # sharping_img = sharping(denoise_img)

  cv2.imshow('img after de-noising and sharping', contrast_stretching_img)

  # edge detection
  edge_img = edge_detect(contrast_stretching_img) 

  cv2.imshow('edge detection img', edge_img)

  # face detection
  coordinator_list, face_detect_img = face_detect(contrast_stretching_img)

  cv2.imshow('face detection img', face_detect_img)

  # face recognition
  face_recognition_img = face_recognition(coordinator_list, face_detect_img)

  # show result
  # sign_image.configure(image=face_recognition_img)
  # sign_image.image=face_recognition_img
  # label.configure(text='Final image')

  cv2.imshow('face recognition img', face_recognition_img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()


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


