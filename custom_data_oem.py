# -*- coding: utf-8 -*-

# project 3A-D20 - An Archive Automation-Decoder 2.0

# 1 language, 1 image processing algorithm and 3 OEM (OCR Engine Modes)

import cv2
import os
import numpy as np
import pandas as pd
import pytesseract
from matplotlib import pyplot as plt
from datetime import date

# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# noise removal
def remove_noise(image):
    return cv2.medianBlur(image,5)
 
# thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)

# skew correction
def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

# count function 
def count(str1, str2):  
    c, j = 0, 0
      
    # loop executes till length of str1 and  
    # stores value of str1 character by character  
    # and stores in i at each iteration. 
    for i in str1:     
          
        # this will check if character extracted from 
        # str1 is present in str2 or not(str2.find(i) 
        # return -1 if not found otherwise return the  
        # starting occurrence index of that character 
        # in str2) and j == str1.find(i) is used to  
        # avoid the counting of the duplicate characters 
        # present in str1 found in str2 
        if str2.find(i)>= 0 and j == str1.find(i):  
            c += 1
        j += 1
    return c

def read_image_text(filename):
    image_text = open(os.path.join('test/ground_truth', os.path.splitext(filename)[0]+".txt"), "r", encoding="utf-8")
    return str(image_text.read())


def plot_image(image, multiple = 'false'):
    if multiple == 'false':
        # split image into channels and rejoin them to display the original image
        b,g,r = cv2.split(image)
        rgb_img = cv2.merge([r,g,b])
    
        # plot the image and display it
        plt.figure(figsize=(4,6), dpi=80)
        plt.imshow(rgb_img)
        plt.title('IMAGE '+os.path.splitext(filename)[0])
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                    hspace = 0, wspace = 0)
        plt.margins(0,0)
        plt_path = 'results/plots/'+os.path.splitext(filename)[0]+'_plot'+'.png'
        plt.savefig(plt_path, bbox_inches = 'tight', pad_inches = 0.1)
        plt.show()
    else:
        # Plot images after preprocessing
        fig = plt.figure(figsize=(12,6))
        ax = []
        rows = 1
        columns = 3
        keys = list(image_types.keys())
        for i in range(rows*columns):
            ax.append( fig.add_subplot(rows, columns, i+1) )
            ax[-1].set_title('Image - ' + keys[i]) 
            plt.imshow(image_types[keys[i]], cmap='gray')
            plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
            plt.margins(0,0)
            
        plt_path = 'results/plots/'+os.path.splitext(filename)[0]+'_plot_ip'+'.png'
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
        plt.margins(0,0)
        plt.savefig(plt_path, bbox_inches = 'tight', pad_inches = 0.1)
        plt.show()

def transcript_images(langs, image_types, words, oem):
    df = pd.DataFrame(columns=['Image name','Translated text','Ground truth', 'Translation language', 'Image preprocessing','Text_TT','Text_GT','diff_TT/GT','match','oem','similarity'])
    is_cw = 0
    temp_cl = 0
    for i in langs:
        # append language to custom config for tesseract
        c_config=r"-l "+i+' --oem '+oem+' '+custom_config
        # print the selected image and the tesseract translation
        for j in image_types:
            print('\n-----------------------------------------')
            print('TESSERACT OUTPUT --> '+str(j))
            print('-----------------------------------------')
            tesseract = pytesseract.image_to_string(image_types[j], config=c_config)
            print(tesseract)
            words.append(tesseract.lower())
            match = ''
            if(temp_cl < count(tesseract.lower(), image_text.lower())):
               temp_cl = count(tesseract.lower(), image_text.lower())
            
            if tesseract.lower().strip() == image_text.lower().strip():
                match = 'Yes'
                is_cw = 1
            else:
                match = 'No'
            # save the data into pandas dataframe
            new_row = {'Image name': filename, 'Translated text': tesseract,'Ground truth': image_text, 'Translation language': i, 'Image preprocessing': j, 'Text_TT': len(tesseract),'Text_GT': len(image_text),'diff_TT/GT': str(round((len(tesseract)/len(image_text))*100, 2))+'%','match': match,'oem': oem, 'similarity': ''}
            df = df.append(new_row, ignore_index=True)
            
                
            

    
    return is_cw, temp_cl, df, words

def init_lists(array):
    orig_list = {}
    correct_list = {}
    acc_list = {}
    
    for i in array:
        orig_list.update({i: 0})
        correct_list.update({i: 0})
        acc_list.update({i: 0})
        
    return orig_list, correct_list, acc_list

def get_accuracy(array, full_list, acc_list, perc_list):
    for i in array:
       perc_list[i] = str(round((acc_list[i]/full_list[i])*100, 2))+'%'
    
    return full_list, acc_list, perc_list

"""
USED LANGUAGES:

    "eng" - English

    "enm" - Enlish, Middle (1100-1500)

    "ita" - Italian

    "ita_old" - Italian-Old

    "lat" - Latin
"""
# directory for getting test images
directory = r'test'
# directory to save the csv file in
result_dir = 'results'
# selected languages for translation
langs = ["enm"]
# custom config for tesseract api
custom_config = r' -c tessedit_char_whitelist="AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz " --psm 6' 
# initialize new pandas dataframe to save data in
df = pd.DataFrame(columns=['Image name','Translated text','Ground truth', 'Translation language', 'Image preprocessing','Text_TT','Text_GT','diff_TT/GT','match','oem','similarity'])
df2 = pd.DataFrame(columns=['Date','Total Word Count','Correct Word Count','Word Accuracy','Total Letter Count', 'Correct Letter Count','Letter Accuracy', 'oem'])
df4 = pd.DataFrame(columns=['Word', 'Count', 'oem'])
# loop through all files in the selected directory

image_types_list = ["Deskewed"]
oem = ["0", "1", "2", "3"]
    
for o in oem:
    words = []
    word_count = {}
    total_word_count = 0
    correct_word_count= 0
    total_letter_count= 0
    correct_letter_count= 0
        
    for filename in os.listdir(directory):
        # only perform tranlsation on image files inside the directory
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".tiff"):
            # get the path to ground truth and save the text into a variable
            image_text = read_image_text(filename)
            
            # read image with opencv
            image = cv2.imread(os.path.join(directory, filename))
        
            plot_image(image)
        
            # Preprocess image 
            gray = get_grayscale(image)
            thresh = thresholding(gray)
            deskewed = deskew(remove_noise(thresh))
            
            image_types = {"Deskewed": deskewed}
            
            is_cw = 0
            temp_cl = 0
            total_word_count += 1
            total_letter_count += len(image_text)
            # loop trought selected languages and use tesseract for each language to recognize text in images
                
            is_cw, temp_cl, data, words = transcript_images(langs, image_types, words, o)
            df = df.append(data, ignore_index=True)
            if is_cw == 1:
                correct_word_count += 1
            
            correct_letter_count += temp_cl
            
        else:
            continue
    
    for word in [ele for ind, ele in enumerate(words,1) if ele not in words[ind:]]:
        word_count[word] = words.count(word)
    
    word_count_list = {}
    
    for key in sorted(word_count.keys()):
        word_count_list[key] = word_count[key] 
    
    # save all the data to csv file
    
    for i in word_count_list:
        new_row = {"Word": i, "Count": word_count[i], 'oem': o}
        df4 = df4.append(new_row, ignore_index=True)
    
    new_row = {'Date': date.today(),'Total Word Count': total_word_count, 'Correct Word Count': correct_word_count,'Word Accuracy': str(round((correct_word_count/total_word_count)*100, 2))+'%','Total Letter Count': total_letter_count, 'Correct Letter Count': correct_letter_count,'Letter Accuracy': str(round((correct_letter_count/total_letter_count*100), 2))+'%', 'oem': o}
    df2 = df2.append(new_row, ignore_index=True)
while True:        
    new_data = input("Do you want to overwrite existing data or append new data? (o=overwrite, a=append): ")
    if new_data is "o" or new_data is "a":
        break
    else:
        print("Please input a correct response")
        continue
# overwrite the data or append the data to the csv file
if new_data == 'o':
    df.to_csv(r'results/translations_oem.csv', index=False, header=True)
    df2.to_csv(r'results/accuracy_oem.csv', index=False, header=True)
    df4.to_csv(r'results/word_count_oem.csv', index=False, header=True)
    print('Data has been overwritten')
elif new_data == 'a':
    df.to_csv(r'results/translations_oem.csv', mode='a', index=False, header=True)
    df2.to_csv(r'results/accuracy_oem.csv', mode='a', index=False, header=True)
    df4.to_csv(r'results/word_count_oem.csv', mode='a', index=False, header=True)
    print('Data has been appended')
    