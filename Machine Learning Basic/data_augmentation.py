import os 
import imutils 
import cv2 
import skimage
from imutils import build_montages
import numpy as np 


class DataAugmentation:

    def __init__(self, image_size, flip = False, smooth =  True,scale = False, rotate=True, contrast  = True):

        self.image_size = image_size
        self.scale  = scale
        self.flip = flip
        self.smooth = smooth
        self.rotate = rotate
        self.contrast = contrast


    def flip_img(self,img):

        '''
        flip image 
        '''
        return np.flip(img)

    def smooth_img(self, img):
        '''
        add noise gause to images
        '''
        return cv2.GaussianBlur(img,(3,3),0)

    def scale_img(self, img, beta):
        return skimage.transform.rescale(img, scale=beta, mode='constant')

    def rotate_img(self, img, angle):

        '''
        rotate image with angle degree 
        '''
        #heigt and with
        (h, w) = img.shape[:2]
        # calculate the center of the image
        center = (w / 2, h / 2)
        scale = 1.0
        M = cv2.getRotationMatrix2D(center, angle, scale)
        img_rotated = cv2.warpAffine(img, M, (h, w))

        return img_rotated
    
    def adjust_contrast_img(self, img, anpha):
        
        '''
        down and up contras of image with anpha > 1: up else down
        '''
        lookUpTable = np.empty((1,256), np.uint8)
        for i in range(256):
            lookUpTable[0,i] = np.clip(pow(i / 255.0, anpha) * 255.0, 0, 255)
        res = cv2.LUT(img, lookUpTable)

        return res 

    def gen_images(self, img_path):

        img = cv2.imread(img_path)
        img = cv2.resize(img, (self.image_size, self.image_size))
        arr_img_gen = []
        arr_img_gen.append(img)
        if self.scale == True:
            arr_img_gen.append(self.scale_img(img, 0.75))
            arr_img_gen.append(self.scale_img(img, 0.5))
            arr_img_gen.append(self.scale_img(img, 1.5))
            arr_img_gen.append(self.scale_img(img, 2))
        if self.smooth == True:
            arr_img_gen.append(self.smooth_img(img))
        if self.rotate ==  True:
            # arr_img_gen.append(self.rotate_img(img, 45))
            arr_img_gen.append(self.rotate_img(img, 90))
            # arr_img_gen.append(self.rotate_img(img, 135))
            arr_img_gen.append(self.rotate_img(img, 180))
            # arr_img_gen.append(self.rotate_img(img, 225))
            arr_img_gen.append(self.rotate_img(img, 270))
        if self.scale == True:

            arr_img_gen.append(self.scale(img, 0.5))
            arr_img_gen.append(self.scale(img, 0.75))
            arr_img_gen.append(self.scale(img, 1.5))
            arr_img_gen.append(self.scale(img, 2.0))

        if self.contrast == True:

            arr_img_gen.append(self.adjust_contrast_img(img, 0.6))
            arr_img_gen.append(self.adjust_contrast_img(img, 0.75))
            arr_img_gen.append(self.adjust_contrast_img(img, 2))
            arr_img_gen.append(self.adjust_contrast_img(img, 1.5))

        if self.flip == True:
            arr_img_gen.append(self.flip_img(img))

        return arr_img_gen
    
    def save(self,folder, file_path_raw):
        '''
        save all file to folder
        '''
        if os.path.isdir(folder) == False:
            os.mkdir(folder)

        list_img_gen = self.gen_images(file_path_raw)
        name_file_raw =  file_path_raw.split('.')[0]
        for (i, img) in enumerate(list_img_gen):

            save_name = name_file_raw + "gen_" + str(i) + ".jpg" 
            
            cv2.imwrite(os.path.join(folder, save_name), img)


if __name__ == "__main__":

    img_path = 'F:/FlyCam Road Segment/dataset/images/A (11).jpg'

    agu =  DataAugmentation(224)
    list_img = agu.gen_images(img_path)

    # agu.save(list_img , "")
    montage = build_montages(list_img, (224, 224), (3, 4))[0]
    title = "Result_gen"
    cv2.imshow(title, montage)
    cv2.waitKey(0)