import os
from PIL import Image
import torchvision.transforms as T

"""
params: filename - string
** please save mask image into ./static
returns: accuracy - int
         predicted mask image file name - string
"""
def infer_image(filename):
    print("inference.py : infer_image() : gets a filename of",filename)


    cropping("./images/", filename)

    return 0.7,"Milkyboki_placeholder.jpg"


def crop_imageUI(input_path, save_path, img_name, image_size=1000, crop_size=500):
    '''
    input_path = "./"
    save_path = "./cd"
    '''
    if ".png" not in img_name:
        img_name += ".png"
        
    if crop_size == 250 and "center" in img_name:
        return
    img_path = input_path + img_name
    orig_img = Image.open(img_path)
    (top_left, top_right, bottom_left, bottom_right, center) = T.FiveCrop(size=(crop_size, crop_size))(orig_img)
    cropped = [top_left, top_right, bottom_left, bottom_right, center]
    if crop_size == 250:
        cropped = [top_left, top_right, bottom_left, bottom_right]
    save_name = img_name.split(".")[0]
    for img_idx in range(len(cropped)):
        name = ""
        if img_idx == 0:                
            name = "top_left"
            if crop_size == 250:
                name = "0"
        elif img_idx == 1:
            name = "top_right"
            if crop_size == 250:
                name = "1"
        elif img_idx == 2:
            name = "bottom_left"
            if crop_size == 250:
                name = "2"
        elif img_idx == 3:
            name = "bottom_right"
            if crop_size == 250:
                name = "3"
        elif img_idx == 4:
            if crop_size == 500:
                name = "center"
        cropped[img_idx].save(save_path +"/" + save_name + "_" +name+".png")
    return


def cropping(input_path, img_name):
    save_folder_500 = "crop_ui_500"
    save_path_500 = input_path + save_folder_500
    os.makedirs(save_path_500, exist_ok=True)
    crop_imageUI(input_path, save_path_500, img_name, image_size=1000, crop_size=500)
    
    
    fileNames = [i[:-4] for i in sorted(os.listdir(save_path_500))]
    save_folder_250 = input_path+"crop_ui_250"
    os.makedirs(save_folder_250, exist_ok=True)
    for i in range(len(fileNames)):
        crop_imageUI(save_path_500+"/", save_folder_250, fileNames[i], image_size=500, crop_size=250)
    return