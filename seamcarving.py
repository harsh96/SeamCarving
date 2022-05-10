# CS6475 - Spring 2022

import numpy as np
import scipy as sp
import cv2
import scipy.signal                     # option for a 2D convolution library
from matplotlib import pyplot as plt    # for optional plots

import copy

'''
----------DEBUGGING HELPER FUNCTIONS----------------
'''

def show_float_image(image):
    #This method is used for debugging intermediate results
    print(image.shape)
    norm_uint8_image = 255*((image-image.min())/(image.max()-image.min()))
    plt.imshow(norm_uint8_image,cmap = 'gray')
    plt.pause(0)

def print_percentage_complete(current, total):
    #This method is used for printing percentage completed while debugging
    per = int(((current+1)*100)/total)
    print("Percentage Completed: "+str(per),end='\r')

'''
----------------------------------------------------
'''


'''
----------ENERGY COST MATRIX FUNCTIONS----------------
'''

def get_backward_energy_cost_matrix(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)
    Ix = cv2.Sobel(src=gray_image, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3)
    Iy = cv2.Sobel(src=gray_image, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3)
    energy_map = abs(Ix) + abs(Iy)

    # energy_map = abs(cv2.Sobel(src=gray_image, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=3))
    # show_float_image(energy_map)    

    M = np.copy(energy_map)
    height, width = M.shape
    for h in range(1, height):
        M[h,1:width-1] += np.min(np.array([np.roll(M[h-1],1), M[h-1], np.roll(M[h-1],-1)]), axis = 0)[1:width-1]
        M[h,0] += min(M[h-1,0], M[h-1,1])
        M[h,width-1] += min(M[h-1,width-2], M[h-1,width-1])
        # M[h,0] += min(M[h-1,0], M[h-1,1])
        # for w in range(1, width-1):
        #     M[h,w] += min(M[h-1,w-1],M[h-1,w], M[h-1,w+1])
        # M[h,width-1] += min(M[h-1,width-2], M[h-1,width-1])

    # show_float_image(M)

    return M


def get_forward_energy_cost_matrix(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)
    M = np.zeros(gray_image.shape)

    I1 = np.roll(gray_image, 1, axis=0) #I(i-1,j)
    I2 = np.roll(gray_image, 1, axis=1) #I(i,j-1)
    I3 = np.roll(gray_image, -1, axis=1) #I(i,j+1)

    CU = abs(I3-I2) # |I(i,j+1) - I(i,j-1)|
    CL = CU + abs(I1-I2) # |I(i,j+1) - I(i,j-1)| + |I(i-1,j) - I(i,j-1)|
    CR = CU + abs(I1-I3) # |I(i,j+1) - I(i,j-1)| + |I(i-1,j) - I(i,j+1)|

    for i in range(1, gray_image.shape[0]):
        # M[i] = min(M[i-1,j-1]+CL[i,j], M[i-1,j]+CU[i,j], M[i-1,j+1]+CR[i,j] )
        M[i] = np.min(np.array([np.roll(M[i-1],1)+CL[i], M[i-1]+CU[i], np.roll(M[i-1],-1)+CR[i]]), axis=0)
    
    # show_float_image(M)

    return M

'''
----------------------------------------------------
'''

'''
---------------SEAM FUNCTIONS-----------------------
'''


def get_vertical_seam(cost_matrix):
    height,width = cost_matrix.shape
    seam_indices = [np.argmin(cost_matrix[height-1])] 
    index_count = 0
    for h in range(height-2,-1,-1):
        w = seam_indices[index_count]
        if w == 0:
            seam_indices.append(np.argmin(cost_matrix[h,0:2]))
        elif w == width-1:
            seam_indices.append(width - 2 + np.argmin(cost_matrix[h,width-2:width]))
        else:
            seam_indices.append(w - 1 + np.argmin(cost_matrix[h,w-1:w+2]))
        index_count += 1

    seam_indices.reverse()
    return seam_indices


def paint_vertical_removal_seam(image, seam_indices, index_tracker):
    for h in range(image.shape[0]):
        w = index_tracker[h,seam_indices[h]]
        image[h,w] = [0,0,255]
    

def remove_vertical_seam(image, seam_indices):
    if len(image.shape) == 3:
        output = np.zeros([image.shape[0],image.shape[1]-1,image.shape[2]],dtype=image.dtype)
    else:
        output = np.zeros([image.shape[0],image.shape[1]-1],dtype=image.dtype)

    for h in range(image.shape[0]):
        w = seam_indices[h]
        output[h,:w] = image[h,:w]
        output[h,w:] = image[h,w+1:]
    return output


def insert_vertical_seams(image, seams_list, insertion_count, redSeams = False):
    seams = len(seams_list[0])
    height, width, channels = image.shape
    seam_insertion_image = np.zeros([height, width+seams*insertion_count, channels])

    for h in range(height):
        seams_list[h].sort()
        seam_insertion_image[h, :seams_list[h][0] + 1] = image[h, :seams_list[h][0] + 1]
        for w in range(seams):

            index = seams_list[h][w]
            insertion_image_index = index + w*insertion_count + 1

            if w == seams-1:
                pixel = image[h,index]
                if redSeams:
                    pixel = [0,0,255]
                seam_insertion_image[h,insertion_image_index:insertion_image_index+insertion_count] = pixel
                seam_insertion_image[h,insertion_image_index+insertion_count:] = image[h, index+1:]
            else:
                pixel = image[h,index]/2.0 + image[h,index+1]/2.0
                if redSeams:
                    pixel = [0,0,255]
                seam_insertion_image[h,insertion_image_index:insertion_image_index+insertion_count] = pixel
                seam_insertion_image[h,insertion_image_index+insertion_count:seams_list[h][w+1]+(w+1)*insertion_count + 1] = image[h, index+1:seams_list[h][w+1]+1]

        # print_percentage_complete(h,height)
    return seam_insertion_image.astype(image.dtype)


def image_vertical_seam_removal(image, seams, cost_function, redSeams=False):

    seam_removal_image = np.copy(image)
    red_seam_image = np.copy(image)
    index_tracker = np.array([np.arange(image.shape[1])]*image.shape[0])

    for seam in range(seams):
        cost_matrix = cost_function(seam_removal_image)
        seam_indices = get_vertical_seam(cost_matrix)
        seam_removal_image = remove_vertical_seam(seam_removal_image, seam_indices)

        if redSeams:
            paint_vertical_removal_seam(red_seam_image, seam_indices, index_tracker)
            index_tracker = remove_vertical_seam(index_tracker, seam_indices)

        # print_percentage_complete(seam,seams)

    return seam_removal_image, red_seam_image


def image_vertical_seam_insertion(image, seams, cost_function, insertion_count=1, redSeams=False):
    seam_removal_image = np.copy(image)
    index_tracker = np.array([np.arange(image.shape[1])]*image.shape[0])

    height, width, channels = image.shape
    
    seams_list = [[] for _ in range(height)]
    
    for seam in range(seams):
        cost_matrix = cost_function(seam_removal_image)
        seam_indices = get_vertical_seam(cost_matrix)
        seam_removal_image = remove_vertical_seam(seam_removal_image, seam_indices)

        for h in range(height):
            x = index_tracker[h,seam_indices[h]]
            seams_list[h].append(x)

        index_tracker = remove_vertical_seam(index_tracker, seam_indices)
        # print_percentage_complete(seam,seams)

    seam_insertion_image = insert_vertical_seams(image, seams_list, insertion_count)

    if redSeams:
        red_seam_image = insert_vertical_seams(image, seams_list, insertion_count, True)
    else:
        red_seam_image = None

    return seam_insertion_image, red_seam_image


'''
----------------------------------------------------
'''

if __name__ == "__main__":

    image = cv2.imread("images/test1.png")

    result, red_result = image_vertical_seam_removal(image, image.shape[1]//2, get_backward_energy_cost_matrix, True)
    cv2.imwrite("images/result1.png", result)
    cv2.imwrite("images/red_result1.png", red_result)

    result, red_result = image_vertical_seam_removal(image, image.shape[1]//2, get_forward_energy_cost_matrix, True)
    cv2.imwrite("images/result2.png", result)
    cv2.imwrite("images/red_result2.png", red_result)

    result, red_result = image_vertical_seam_insertion(image, image.shape[1]//2, get_backward_energy_cost_matrix, 1, True)
    cv2.imwrite("images/insertion-result1.png", result)
    cv2.imwrite("images/insertion-red_result1.png", red_result)

    result, red_result = image_vertical_seam_insertion(image, image.shape[1]//2, get_forward_energy_cost_matrix, 1, True)
    cv2.imwrite("images/insertion-result2.png", result)
    cv2.imwrite("images/insertion-red_result2.png", red_result)
    

