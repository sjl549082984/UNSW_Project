import cv2
import numpy as np
import  os
from matplotlib import pyplot as plt
import argparse

def task1(image):
    threshold = 100
    epst = 0.01
    while 1:
        mL = image[image <= threshold].mean()
        mH = image[image > threshold].mean()
        t_new = (mL + mH) / 2
        if abs(threshold - t_new) < epst:
            break
        threshold = t_new
        print("threshold =",threshold)
    max_image = np.zeros((image.shape[0], image.shape[1]), dtype='uint8')
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i, j] >= threshold:
                max_image[i, j] = 255
            else:
                max_image[i, j] = 0
    median = cv2.medianBlur(max_image, 3)
    median = cv2.medianBlur(median, 3)
    final = cv2.medianBlur(median, 3)
    return threshold, final




def CCL(f, offsets, reverse):
    rows, cols = f.shape
    label_idx = 0
    if reverse == False:
        rows_temp = [0, rows, 1]
    else:
        rows_temp = [rows - 1, -1, -1]
    if reverse == False:
        cols_temp = [0, cols, 1]
    else:
        cols_temp = [cols-1, -1, -1]
    for row in range(rows_temp[0], rows_temp[1], rows_temp[2]):
        for col in range(cols_temp[0], cols_temp[1], cols_temp[2]):
            label = 256
            if f[row][col] < 1:
                continue
            for offset in offsets:
                neighbor_row = min(max(0, row+offset[0]), rows-1)
                neighbor_col = min(max(0, col+offset[1]), cols-1)
                value = f[neighbor_row, neighbor_col]
                if value < 1:
                    continue
                label = value if value < label else label
            if label == 255:
                label_idx += 1
                label = label_idx
            f[row][col] = label
    return f



def task2(binary_img):
    linked = []
    f = binary_img
    labels = np.zeros((f.shape[0], f.shape[1]), dtype='uint8')
    OFFSETS_8 = [[-1, -1], [0, -1], [1, -1],
                 [-1, 0], [0, 0], [1, 0],
                 [-1, 1], [0, 1], [1, 1]]
    label = 1
    linked.append(0)
    linked.append(1)
    for i in range(f.shape[0]):
        if i == 0:
            continue
        for j in range(f.shape[1]):
            if j == 0:
                continue
            neighbors = []
            if f[i, j] != 0:
                if f[i-1][j] > 1 :
                    neighbors.append(f[i-1][j])
                if f[i][j-1] > 1 :
                    neighbors.append(f[i][j-1])

                if not neighbors:
                    label = label + 1
                    linked.append(label)
                    labels[i, j] = label
                    linked[label] = label
                else:
                    smallestLabel = min(neighbors)
                    labels[i, j] = smallestLabel
    binary_img = CCL(binary_img, OFFSETS_8, False)
    binary_img = CCL(binary_img, OFFSETS_8, True)
    binary_img = CCL(binary_img, OFFSETS_8, False)
    binary_img = CCL(binary_img, OFFSETS_8, True)

    re_map = []
    number = []
    rows, cols = binary_img.shape
    for row in range(rows):
        for col in range(cols):
            var = binary_img[row][col]
            if var < 0.5:
                continue
            if var in re_map:
                location = re_map.index(var)
                num = location + 1
            else:
                location = len(re_map)
                num = location + 1
                re_map.append(var)

            binary_img[row][col] = num
            if num not in number:
                number.append(num)

    return binary_img, len(number)

def task3(f, min_area):
    compare = {}
    sum =0
    rows, cols = f.shape
    for row in range(rows):
        for col in range(cols):
            var = f[row][col]
            if var not in compare:
                compare[var] = 0
            else:
                compare[var] = compare[var] +1
    result_max = max(compare,key=lambda x:compare[x])
    result_min = min(compare, key=lambda x: compare[x])
    del compare[result_max]
    del compare[result_min]

    for k, v in compare.items():
        sum = sum + v
    average = sum / len(compare)
    #print(average)

    pixel =[]
    for k, v in compare.items():
        if v > min_area:
            pixel.append(k)
    for row in range(rows):
        for col in range(cols):
            var = f[row][col]
            if var not in pixel:
                f[row][col] = 0
            else:
                f[row][col] = 255
    return f , len(pixel) / len(compare)



my_parser = argparse.ArgumentParser()
my_parser.add_argument('-o','--OP_folder', type=str,help='Output folder name', default = 'OUTPUT')
my_parser.add_argument('-m','--min_area', type=int,action='store', required = True, help='Minimum pixel area to be occupied, to be considered a whole rice kernel')
my_parser.add_argument('-f','--input_filename', type=str,action='store', required = True, help='Filename of image ')
# Execute parse_args()
args = my_parser.parse_args()

print(args.OP_folder)

if args.input_filename:
    file_name =args.input_filename
else:
    file_name = 'rice_img1.png'

if args.min_area:
    min_area =args.min_area
else:
    min_area = 100

if args.OP_folder:
    path = args.OP_folder
else:
    path = 'OUTPUT'

img1 = cv2.imread(file_name, 0)

threshold ,img2 = task1(img1)

# Q1
text = 'Threthold Value = '+str(threshold)
plt.imshow(img2,cmap = 'gray'),plt.title(text)
plt.xticks([]), plt.yticks([])
question1_name = file_name[0:-4]+'_Task1.png'
plt.savefig(path +'\\'+question1_name)

# Q2
path = '.\\' +path
binary_img, points = task2(img2)
print("number of  kernels:", points)
question2_name = file_name[0:-4]+'_Task2.png'
cv2.imwrite(os.path.join(path , question2_name), binary_img)

# Q3
binary_img, number = task3(binary_img, min_area)
print("percentage of damage kernels:", number)
question3_name = file_name[0:-4]+'_Task3.png'
cv2.imwrite(os.path.join(path , question3_name), binary_img)
cv2.waitKey(-1)

