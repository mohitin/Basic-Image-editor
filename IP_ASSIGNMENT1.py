#--------------------------------------BASIC IMAGE EDITOR-------------------------------------------------------
#Importing libraries
from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog
import numpy as np
from IPython.display import display, Math, Latex
import math
import os
#creating tkinter framework
root = Tk()
root.title('Image Processing')
root.configure(bg='#93FFE8')
root.geometry('1500x1500')
# here r is initialized for using in buttons for operation
r = IntVar()
r.set("0")
# Global variable are defined to be used further
global my_image
global pathh, image_path,type, list,img
global prev, current
list=[]
##Functions:
#----------------------Display function for all type of images----------------------------------
def Display(arr):
    global image_path,list
    #imag = Image.open(pathh)
    canva.delete("all")
    if (type=="color"):
        imag1 = Image.fromarray(arr, 'HSV')
        #imag1 = Image.fromarray(arr)
    else:
         imag1 = Image.fromarray(arr,'L')
    imag1.thumbnail((400, 400))
    imag2 = ImageTk.PhotoImage(imag1)
    canva.create_image(450, 155, image=imag2)
    canva.image = imag2
def undo():
    Display(img)
    
#-------------------Function to select file from local directory--------------------------------------
def open():
    global pathh,image_path,type,list,img
    pathh=filedialog.askopenfilename()                   
    image_path=Image.open(pathh)
    image_path.thumbnail((1000,700))
    if(image_path.mode=="RGB" or image_path.mode=="RGBA" or image_path.mode=="CMYK"or image_path.mode=="LAB"or image_path.mode=="HSV" ):
        type="color"
        img = np.array(image_path.convert('HSV'))
    else:
        type="grey"
        img=np.array(image_path.convert('L'))
    list.clear()
    list.append(img)
    #pix = np.array(image_path)
    Display(img)    
#------------------------Histogram Equalization---------------------------------------------------
def histogramEqualization():
    #coverting image into array
    img = np.asarray(image_path)
    #flattening image to one dimension
    flattening = img.flatten()
    # creating own histogram function
    def get_histogram(image_path, bins):
        # array with size of bins, set to zeros
        histogram = np.zeros(bins)
        # loop through pixels and sum up counts of pixels
        for pixel in image_path:
            histogram[pixel] += 1
        # return our final result
        return histogram
    hist = get_histogram(flattening, 256)
    # create our cumulative sum function
    def cumulative(input):
        input = iter(input)
        b = [next(input)]
        for i in input:
            b.append(b[-1] + i)
        return np.array(b)

    # execute the fn
    cd = cumulative(hist)
    # # numerator & denomenator
    nk = (cd - cd.min()) * 255
    N = cd.max() - cd.min()

    # # re-normalize the cdf
    cs = nk / N
    # coverting into uint8 since we can't use floating point values in images
    cs = cs.astype('uint8')
    # get the value from cumulative sum for every index in flat, and set that as img_new
    img_new = cs[flattening]
    # unflattening back
    imgNew = np.reshape(img_new, img.shape)
    # display output
    Display(imgNew)
# -----------------------------------Gamma Correction-----------------------------------------------
def gammaCorrection():
    global list,type,gamma
    img = np.asarray(image_path)
    #entry level to take an input
    label_value=Label(root,text='Enter Value of Gamma:',bg='#93FFE8')
    label_value.place(x=200,y=235)
    e1=Entry(root,width=3,bg='#40E0D0')
    e1.place(x=350,y=237)   
    # command after pressing confirm button
    def sub_gamma():
        global gamma
        gamma = float(e1.get())
        #Program for Gamma Correction:
        transformed = list[-1]
        #flattening for performing operation
        if (type == "color"):
            a = np.rollaxis(transformed, -1)[2]
        else:
            a = transformed
        #main algorithm
        c = 255 / math.pow(255,gamma)
        a = c * (np.power(a,gamma))
        a = np.array(np.round(a))
        #unflattening back
        if (type == "color"):
            transformed[:, :, 2] = a
        else:
            transformed = a
        list.append(transformed)
        Display(transformed)
        #vanishing buttons and entries
        label_value.after(100, label_value.destroy)
        e1.after(100, e1.destroy)
        confirm.after(50, confirm.destroy)
    confirm = Button(root, text='Confirm',command=sub_gamma,bg='#2D90D5')
    confirm.place(x=400,y=232)
#----------------lOGORITHMIC TRANSFORMATION---------------------------------------------------    
def Log():
    global list, type
    img = np.asarray(image_path)
    img2 = list[-1]
    #flattening
    if (type == "color"):
        a = np.rollaxis(img, -1)[2]
    else:
        a = img2
    #main algorithm
    c=255/np.log(1+255)
    print(c)
    a=c*(np.log(a+1))
    a=np.array(np.round(a))
    #unflattening back
    if (type == "color"):
        img2[:, :, 2] = a
    else:
        img2 = a
    list.append(img2)
    Display(img2)
#-----------------Negative of a image--------------------------------------------------
def Negative():
    global img
    global list, type
    img2 = list[-1]
    #flattening
    if (type == "color"):
        a = np.rollaxis(img2, -1)[2]
    else:
        a = img2
    #main algorithm
    negative=255-a
    if (type == "color"):
        img2[:, :, 2] = negative
    else:
        img2 = negative
    list.append(img2)
    Display(img2)
#------------------------Smoothening(Blurring) of Image--------------------------------------------
def Smoothening():
    global img
    label_value=Label(root,text='Enter Value of Sigma:',bg='#93FFE8')
    label_value.place(x=180,y=295)
    e2=Entry(root,width=3,bg='#40E0D0')
    e2.place(x=320,y=297)
    confirm = Button(root, text='Confirm',bg='#2D90D5')
    confirm.place(x=370,y=293)
    def convolution(input_image, kernel):
        #height and width of an image
        h = input_image.shape[0] 
        w = input_image.shape[1] 

        # height and width of an kernal
        k_h = kernel.shape[0]
        k_w = kernel.shape[1]
        ## Condition to check whether the input image is color or grayscale then  padding
        if(len(input_image.shape) == 3):
            pad = np.pad(input_image, pad_width=((k_h // 2,k_h // 2),(k_w // 2, k_w // 2),(0,0)), mode='constant',constant_values=0).astype(np.float32)
        elif(len(input_image.shape) == 2):
            pad = np.pad(input_image, pad_width=((k_h // 2,k_h // 2),(k_w // 2,k_w // 2)), mode='constant', constant_values=0).astype(np.float32)

        # floor division of dimension of an kernal
        h1 = k_h // 2
        w1 = k_w // 2
        # padding zeros to image
        image_convolution = np.zeros(pad.shape)
        # Main convolution step
        for i in range(h1, pad.shape[0]-h1):
            for j in range(w1, pad.shape[1]-w1):
                x = pad[i-h1:i-h1+k_h, j-w1:j-w1+k_w]
                x = x.flatten()*kernel.flatten()
                image_convolution[i][j] = x.sum()
        h_end = -h1
        w_end = -w1
        # returning the convolved image
        if(h1 == 0):
            return image_convolution[h1:,w1:w_end]
        if(w1 == 0):
            return image_convolution[h1:h_end,w1:]
        return image_convolution[h1:h_end,w1:w_end]
    #main body
    
    #reading image
    sigma=e2.get()
    #image = Image.open(image_path)
    image = np.asarray(image_path)
    # Common formula to decide filter size
    size = 2 * int(4 * sigma + 0.5) + 1
    #padding zeros
    gaussian = np.zeros((size,size), np.float32)
    m = size//2
    n = size//2
    # Formula : g= 1/(2pi(sigma)^2)*e-(x^2+y^2)/2(sigma)^2 is implemeted below
    for x in range(-m, m+1):
        for y in range(-n, n+1):
            x1 = 2*np.pi*(sigma**2)
            x2 = np.exp(-(x**2 + y**2)/(2* sigma**2))
            gaussian[x+m, y+n] = (1/x1)*x2
    #returns the array of zeros of same shape and type
    im_filtered = np.zeros_like(a, dtype=np.float32) 
    #convolution with original image
    for c in range(3):
        im_filtered[:, c] = convolution(a[:, c], gaussian)
    # returns final filtered image
    convolved_image1=im_filtered.astype(np.uint8)
    convolved_image=np.array(convolved_image1)
    if (type == "color"):
        img[:, :, 2] = convolved_image
    else:
        img = convolved_image
    #list.append(img)
    Display(img)
# #------------------------------SHARPENING OF AN IMAGE----------------------------------------------
def Sharpen():
    label_value=Label(root,text='Enter Value of Sigma and Alpha:',bg='#93FFE8')
    label_value.place(x=160,y=325)
    e3=Entry(root,width=3,bg='#40E0D0')
    e3.place(x=350,y=329)
    e4=Entry(root,width=3,bg='#40E0D0')
    e4.place(x=379,y=329)
    confirm = Button(root, text='Confirm',bg='#2D90D5')
    confirm.place(x=420,y=324)
    def convolution(input_image, kernel):
        #height and width of an image
        h = input_image.shape[0] 
        w = input_image.shape[1] 

        # height and width of an kernal
        k_h = kernel.shape[0]
        k_w = kernel.shape[1]
        ## Condition to check whether the input image is color or grayscale then  padding
        if(len(input_image.shape) == 3):
            pad = np.pad(input_image, pad_width=((k_h // 2,k_h // 2),(k_w // 2, k_w // 2),(0,0)), mode='constant',constant_values=0).astype(np.float32)
        elif(len(input_image.shape) == 2):
            pad = np.pad(input_image, pad_width=((k_h // 2,k_h // 2),(k_w // 2,k_w // 2)), mode='constant', constant_values=0).astype(np.float32)

        # floor division of dimension of an kernal
        h1 = k_h // 2
        w1 = k_w // 2
        # padding zeros to image
        image_convolution = np.zeros(pad.shape)
        # Main convolution step
        for i in range(h1, pad.shape[0]-h1):
            for j in range(w1, pad.shape[1]-w1):
                x = pad[i-h1:i-h1+k_h, j-w1:j-w1+k_w]
                x = x.flatten()*kernel.flatten()
                image_convolution[i][j] = x.sum()
        h_end = -h1
        w_end = -w1
        # returning the convolved image
        if(h1 == 0):
            return image_convolution[h1:,w1:w_end]
        if(w1 == 0):
            return image_convolution[h1:h_end,w1:]
        return image_convolution[h1:h_end,w1:w_end]
    # Gaussian filtering
    def gaussianFilter(sigma):
        size = 2 * int(4 * sigma + 0.5) + 1
        gaussian = np.zeros((size, size), np.float32)
        m = size//2
        n = size//2

        for x in range(-m, m+1):
            for y in range(-n, n+1):
                x1 = 2*np.pi*(sigma**2)
                x2 = np.exp(-(x**2 + y**2)/(2* sigma**2))
                gaussian[x+m, y+n] = (1/x1)*x2
        return gaussian
    #Note: First derivative of x and y are almost equal to gaussian filter function 
    # function to calcuate first derivative of x
    def firstDerivative_X(sigma):
        size = 2 * int(4 * sigma + 0.5) + 1
        gaussian = np.zeros((size, size), np.float32)
        m = size//2
        n = size//2

        for x in range(-m, m+1):
            for y in range(-n, n+1):
                gaussian[x+m, y+n] = y
        return gaussian
    # function to calcuate first derivative of y
    def firstDerivative_Y(sigma):
        size = 2 * int(4 * sigma + 0.5) + 1
        gaussian = np.zeros((size, size), np.float32)
        m = size//2
        n = size//2

        for x in range(-m, m+1):
            for y in range(-n, n+1):
                gaussian[x+m, y+n] = x
        return gaussian
    #sharpening of an image
#def SharpenImage(image,sigma, alpha):
    sigma=e3.get()
    alpha=e4.get()
    #image = imageio.imread(image)
    im_filtered = np.zeros_like(image_path, dtype=np.float32)
    #convolving the sharpening kernal with an original image
    #all the functions used are described above 
    for c in range(3):
        im_filtered[:, :, c] = convolution(image_path[:, :, c], firstDerivative_X(sigma)*firstDerivative_Y(sigma)* gaussianFilter(sigma))
        output=np.clip((image_path - (alpha * im_filtered)),0,255).astype(np.uint8)
    output1=np.array(output)
    Display(output1)
#-----------Function to define operation on each click-----------------------------------------------
def clicked(value):
        global my_image
        if value==1:
            histogramEqualization()
            #             myLabel = Label(root, text= " HELLO THERE 1")
            #             myLabel.place(x=800,y=200)
        if value==2:
            gammaCorrection()
        if value==3:
            Log()
        if value==4:
            Smoothening()
        if value==5:
            Sharpen()
        if value==6:
            Negative()  
#canvas to display image
canva=Canvas(root,width="900",height="300",relief=RIDGE,bd=5,bg='#87dec3')
canva.place(x=280,y=400)
# --------------------------------Buttons for various required operations------------------------------------------
label0=Label(root,bg='#40E0D0', text='IMAGE PROCESSING (EE610)-Basic Image Editor',font=('arial',15,'bold'))
label0.place(x=500,y=10)
label1=Label(root,text='STEP 2: Select the Operation',font=('arial',15,'bold'),bg='#40E0D0')
label1.place(x=5,y=170)
r1=Radiobutton(root, text="Equalize Histogram",bg='#93FFE8',font=('arial',13), variable=r, value=1, command=lambda: clicked(r.get()))
r2=Radiobutton(root, text="Gamma Correction",bg='#93FFE8',font=('arial',13), variable=r, value=2, command=lambda: clicked(r.get()))
r3=Radiobutton(root, text="Log transformation",bg='#93FFE8',font=('arial',13), variable=r, value=3, command=lambda: clicked(r.get()))
r4=Radiobutton(root, text="Smoothening ",bg='#93FFE8',font=('arial',13), variable=r, value=4, command=lambda: clicked(r.get()))
r5=Radiobutton(root, text="Sharpening ",bg='#93FFE8',font=('arial',13), variable=r, value=5, command=lambda: clicked(r.get()))
r6=Radiobutton(root, text="Negative of a Image",bg='#93FFE8',font=('arial',13), variable=r, value=6, command=lambda: clicked(r.get()))
r1.place(x=5,y=200)
r2.place(x=5,y=230)
r3.place(x=5,y=260)
r4.place(x=5,y=290)
r5.place(x=5,y=320)
r6.place(x=5,y=350)
#labels
label3=Label(root,text='STEP 1: Select the file',font=('arial',15,'bold'),bg='#40E0D0')
label3.place(x=5,y=40)
my_btn = Button(root, text="Click here to select",padx=60,pady=10, command=open,bg='#2D90D5').place(x=615,y=90)
label3=Label(root,text='Output: ',font=('arial',15,'bold'),bg='#40E0D0')
label3.place(x=5,y=410)
exit_btn = Button(root, text="Exit Operation",padx=30,pady=10, command=root.destroy,bg='#2D90D5')
exit_btn.place(x=1295,y=735)
revert_btn = Button(root, text="Undo Operation",padx=30,pady=10,bg='#2D90D5',command=undo)
revert_btn.place(x=30,y=735)
save_btn = Button(root, text="Save Image",padx=30,pady=10, command=root.destroy,bg='#2D90D5')
save_btn.place(x=680,y=735)
root.mainloop()
# ---------------------------------------------------end of the program-----------------------------------------------
