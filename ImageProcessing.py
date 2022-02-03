from PIL.Image import Image
from numpy.core.fromnumeric import shape
from scipy.ndimage import interpolation
import skimage.io 
import matplotlib.pyplot as plt
import numpngw
import Functions
import scipy.ndimage as ndimage
import pandas as pd
import numpy as np
import math
import timeit
import pydicom
from skimage import measure


file =pd.read_excel('Cancer.xlsx')
images = file.to_numpy()

# file_ =pd.read_csv('Mask\\Mask.csv')
# images = file_.to_numpy()
for i in range(len(images)):
# for i in range():
    # # DICOM IMAGES #
    start = timeit.default_timer()    
    #     image= pydicom.read_file('D:\\SBME\\Graduation Project\\Mass-Training_'+str(images[i][2])+'_'+str(images[i][4])+'_'+str(images[i][5])+'.dcm')
    #     image=image.pixel_array
    #     original = image
    #     tumrMask= pydicom.read_file('D:\\SBME\\Graduation Project\\Mass-Training_'+str(images[i][2])+'_'+str(images[i][4])+'_'+str(images[i][5])+'_'+str(images[i][6])+'_mask1.dcm')
    #     tumrMask=tumrMask.pixel_array
    #     print('Mass-Training_'+str(images[i][2])+'_'+str(images[i][4])+'_'+str(images[i][5])+'.dcm')
    #     print('Mass-Training_'+str(images[i][2])+'_'+str(images[i][4])+'_'+str(images[i][5])+'_'+str(images[i][6])+'_mask1.dcm')
    # # if(images[i][7]=="-"):    
    image= skimage.io.imread('D:\\SBME\\Graduation Project\\Mini_DDSM\\Cancer-new\\' +str(images[i+5][1]))  
    print(images[i+5][1])   
    image=Functions.CropBorders(img=image) 
    original =skimage.io.imread('D:\\SBME\\Graduation Project\\Mini_DDSM\\Cancer-new\\' +str(images[i+5][1]))
    original=Functions.CropBorders(img=original)
    if(images[i+5][7]!="-"):     
        tumrMask= skimage.io.imread('D:\\SBME\\Graduation Project\\Mini_DDSM\\archive\\'+str(images[i+5][7]))
        tumrMask=ndimage.binary_fill_holes(tumrMask)
        tumrMask=Functions.CropBorders(tumrMask)
  

    # whiteimage= skimage.io.imread('.\\Mask\\white.png')
    # whiteimage=np.divide(whiteimage,255)
    # whiteimage=np.subtract(whiteimage,1)
    # whiteimage=np.resize(whiteimage,(image.shape[0],image.shape[1]))
    horizontal_flip =Functions.HorizontalFlip(mask=image)
    if horizontal_flip:
        flipped_img = np.fliplr(image)
        image= flipped_img
        flipped_img2 = np.fliplr(original)
        original=flipped_img2  
        if(images[i+5][7]!="-"):       
            flipped_mask=np.fliplr(tumrMask)
            tumrMask = flipped_mask
    cnt=0    
    # for y in range(image.shape[0]) :
    #     for x in range(image.shape[1]):
    #         cnt=cnt+1
    #         if(image[y][x]<=5000):
    #             image[y][x]=0
    # print(cnt)  
    # plt.imshow(image,cmap='gray')
    # plt.show()
    # binarised_img = Functions.OwnGlobalBinarise(img=image, thresh=0.08*image.max(), maxval=1.0)
    histogram, bin_edges = np.histogram(image, bins=256)
    histogram=list(histogram)
    bin_edges=list(bin_edges)
    # print(histogram.index(max(histogram)))
    histogramValue=bin_edges[0:-1][histogram.index(max(histogram))]
    # print(histogramValue)
    
    binarised_imghist = Functions.OwnGlobalBinarise(img=image, thresh=histogramValue*2, maxval=1.0)

    # plt.imshow(binarised_img,cmap='gray')
    # plt.show()    
    
    plt.imshow(binarised_imghist,cmap='gray')
    plt.show()

    # contours = measure.find_contours(binarised_img, 0.8)
    contours = measure.find_contours(binarised_imghist, 0.8)
    # contour=max(contours.shape[0])
    fig, ax = plt.subplots()
    ax.imshow(image, cmap=plt.cm.gray)
    cont=[]
    for contou in contours:
        cont.append(contou.shape[0])
    # print(max(cont))    
    contourX=[]
    contourY=[] 
    for contour in contours:    
        if contour.shape[0]==max(cont):
            for index in range (len(contour)):
                contourY.append(contour[index][0])
                contourX.append(contour[index][1])
            ax.plot(contour[:, 1], contour[:, 0], linewidth=1)
            # print(len(contour))
    contourX=np.array(contourX)    
    contourY=np.array(contourY)    
    contourX=contourX.astype(int)        
    contourY=contourY.astype(int)        
    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()
    contourY2=[]
    contourX2=[]

    for l,k in zip(contourY,contourX):
        if l not in contourY2:
            contourY2.append(l)
            contourX2.append(k)
    # print(len(contourX2))  
    # print(len(contourY2)) 

    index= 0.3*len(contourX2)
    test=[]
    for ii in range(int(index)):
        z = np.polyfit(contourY2[int(ii):int(ii+index)], contourX2[int(ii):int(ii+index)],3)
        p = np.poly1d(z)
        y = p(contourY2[ii])
        test.append(y)
    for ii in range(int(index),len(contourX2)):
        z = np.polyfit(contourY2[int(ii-(index*0.3)):(int(ii-(index*0.3))+int(index))], contourX2[int(ii-(index*0.3)):(int(ii-(index*0.3))+int(index))],3)
        p = np.poly1d(z)
        y = p(contourY2[ii])
        test.append(y)
    plt.plot(contourX2,contourY2,test,contourY2)
    plt.xlim(0,image.shape[1])
    plt.ylim(image.shape[0],0)
    plt.show()

    # print (image.shape) 
    # print(len(test))       
    BlackImage = np.zeros(image.shape, np.uint16)
    Heatmap = np.zeros(image.shape, np.uint16)
    InvertedHeatmap = np.zeros(image.shape, np.uint16)
    for conty in range (len(test)):
        for contx in range (image.shape[1]):
            if contx<test[conty]:
                BlackImage[conty+min(contourY2)][contx]=1
                Heatmap[conty+min(contourY2)][contx]=1
                InvertedHeatmap[conty+min(contourY2)][contx]=1

    plt.imshow(BlackImage,cmap='gray')
    plt.show()
    interpolatedimage=BlackImage*original
    interpolatedimage=interpolatedimage.astype('uint16')
    stop = timeit.default_timer()
    print('Time: ', (stop - start)/60) 
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize = (22, 10))
    ax[0].imshow(original,cmap='gray')
    ax[1].imshow(image,cmap='gray')
    ax[2].imshow(interpolatedimage,cmap='gray')
    plt.show()
    # interpolatedimage=Functions.CropBorders(interpolatedimage)
    # numpngw.write_png('D:\\SBME\\Graduation Project\\Mini_DDSM\\Cancer_Final\\'+str(images[i+5][1]),interpolatedimage) 

    if(images[i+5][7]!="-"):   
        maskcontours = measure.find_contours(tumrMask, 0.8)
        maskcont=[]
        for maskcontou in maskcontours:
            maskcont.append(maskcontou.shape[0])
        maskcontourX=[]
        maskcontourY=[] 
        for maskcontour in maskcontours:
            if maskcontour.shape[0]==max(maskcont):
                print(maskcontour.shape)
                for it in range (len(maskcontour)):
                    maskcontourY.append(maskcontour[it][0])
                    maskcontourX.append(maskcontour[it][1])
        maskCenterX=(max(maskcontourX)+min(maskcontourX))/2      
        maskCenterY=(max(maskcontourY)+min(maskcontourY))/2      
        print((max(maskcontourX)+min(maskcontourX))/2)        
        print((max(maskcontourY)+min(maskcontourY))/2)  
        Diagonal=math.sqrt  (image.shape[0]**2 + image.shape[1]**2)
        for y in range(image.shape[0]):
            for x in range(image.shape[1]): 
                dist=math.sqrt((y-maskCenterY)**2 + (x-maskCenterX)**2)
                r=round((dist/Diagonal),2)
                # print(r)
                if(interpolatedimage[y][x]!=0):
                    Heatmap[y][x]=(1-r)*65535
                    InvertedHeatmap[y][x]=r*65535
        # for y in range(image.shape[0]):
        #     for x in range(image.shape[1]): 
        #         dist=math.sqrt((y-maskCenterY)**2 + (x-maskCenterX)**2)
        #         r=round((dist/Diagonal),2)
        #         # print(r)
        #         if(interpolatedimage[y][x]!=0):
                              
        fig, ax = plt.subplots(nrows=1, ncols=4, figsize = (22, 10))
        ax[3].imshow(Heatmap, cmap=plt.cm.gray)
        ax[2].imshow(InvertedHeatmap, cmap=plt.cm.gray)
        ax[1].imshow(tumrMask, cmap=plt.cm.gray)
        ax[0].imshow(original, cmap=plt.cm.gray)
        plt.show()
        # numpngw.write_png('F:\\Mini-DDSM\\archive\\Heatmap\\C_0004_1.RIGHT_CC.png',interpolatedimage) 
        # numpngw.write_png('F:\\Mini-DDSM\\archive\\Heatmap\\' +str(images[i+5][1]),interpolatedimage) 
        

            