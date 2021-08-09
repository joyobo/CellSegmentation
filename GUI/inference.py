import os
from PIL import Image
from scipy.io import loadmat, savemat
import torch
import torchvision.transforms as T
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from PIL import Image

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)

from model import *
from utils import *
from stitch_images import *

"""
params: filename - string
** please save mask image into ./static
returns: accuracy - int
         predicted mask image file name - string
"""
def delete_files(path):
    test = os.listdir(path)
    for item in test:
        os.remove(os.path.join(path, item))


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]

class CellDataset(Dataset):

    def __init__(self, root_dir, transform=None, training=True):
        self.fileNames = [i[:-4] for i in sorted(os.listdir(root_dir+"/crop_ui_500/Images"))]
        self.root_dir = root_dir
        self.transform = transform
        self.training = training

    def __len__(self):
        return len(self.fileNames)
    
    """
    returns: tuple of
    - image
    - mask of size n x n with unique values/classes ranging from 0 to 4 
        0: background
        1: others (1)
        2: inflammatory(2)
        3: healthy epithelial(3) , dysplastic/malignant epithelial(4)
        4: fibroblast(5) , muscle(6) , endothelial(7)
    """
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, "crop_ui_500/Images", self.fileNames[idx])+".png"

        print("img_name", img_name)
        mask_name = os.path.join(self.root_dir,"crop_ui_500/Labels",self.fileNames[idx])+".mat"
        image = img = Image.open(img_name).convert('RGB')
        x = loadmat(mask_name)['type_map']
        x[(x==3)|(x==4)]=3
        x[(x==5)|(x==6)|(x==7)]=4
        x=np.pad(x.astype(int), 6)       
        # # 250x250 images
        # x=np.pad(x.astype(int),3)

        # 500x500 images
        # x=np.pad(x.astype(int),6)

        # 1000x1000 images
        # x=np.pad(x.astype(int),12)
        if self.transform:
            image = self.transform(image)
        return image, to_categorical(x,5).transpose(2, 0, 1), self.fileNames[idx] #(num_classes=5, n, n)


def infer_image(filename_image, filename_label):
    print("inference.py : infer_image() : gets a filename of",filename_image)

    cropping("./images/", filename_image, filename_label)
    # Initialize model and optimizer
    device = torch.device('cpu')
    model = build_model("resnet")
    model.load_state_dict(torch.load("../best_model_checkpoints/ResNet_Attention_UNet.pth", map_location=device))
    model.eval()


    transform = transforms.Compose([
    # 250x250 images
    # transforms.Pad(3),

    # 500x500 images
    transforms.Pad(6),

    # 1000x1000 images
    # transforms.Pad(12),
    
    transforms.ToTensor()    
])
    root_dir = "./images"
    test_data = CellDataset(root_dir=root_dir, transform = transform, training=False)
    # load test data in batches
    test_loader = DataLoader(test_data, batch_size=4, num_workers=0)

    path = "./images/" + filename_label
    x = loadmat(path)['type_map']
    x[(x==3)|(x==4)]=3
    x[(x==5)|(x==6)|(x==7)]=4
    x=x.astype(int)
    actualFullTarget = torch.from_numpy(to_categorical(x,5).transpose(2, 0, 1))
    
    pred_dict = {}
    for data, target, filenames in test_loader: # this is only for batch size of 1 and size=250
        for i in range(4):
            input = torch.unsqueeze(data[i], 0)
            label = torch.unsqueeze(target[i], 0)
            pred = test(model, device, input, label)
            pred_dict[filenames[i]] = pred

    stitch = stitch_500(pred_dict)
    predFinalMask = printColoredMask(stitch)
    print(predFinalMask.shape)
    print("\nFinal Stitch", predFinalMask.shape)
    
    # im = Image.fromarray(predFinalMask)
    # im.save("./static/predFinalMask.png", 'RGB')
    plt.imshow(predFinalMask)
    plt.savefig("./static/predFinalMask.png")

    acutalMask = printColoredMask(actualFullTarget.numpy())
    plt.imshow(acutalMask)
    plt.savefig("./static/acutalMask.png")
    
    actualFullTarget = torch.unsqueeze(actualFullTarget, 0)
    pq_score, dice_score = evalutation(stitch, actualFullTarget)


    delete_files("./images/crop_ui_500/Images/")
    delete_files("./images/crop_ui_500/Labels/")

    return pq_score, dice_score, "predFinalMask.png", "acutalMask.png"




"""
0: black: background
1: red: others (1)
2: green: inflammatory(2)
3: dark blue: healthy epithelial(3) , dysplastic/malignant epithelial(4)
4: light blue: fibroblast(5) , muscle(6) , endothelial(7)
"""
# params: 5 x n x n numpy or n x n x 5
def printColoredMask(npMask, numchannel=5):
    if npMask.shape[-1]!=5:
        npMask=npMask.transpose(1, 2, 0)
    finalnpMask=np.where(npMask[:,:,1]==1,255,0) # one color
    finalnpMask=finalnpMask[:,:,None]
    temp=np.where((npMask[:,:,2]==1)|(npMask[:,:,4]==1),255,0) # one color
    finalnpMask = np.concatenate((finalnpMask,temp[:, :, None]),axis=2)
    temp=np.where((npMask[:,:,3]==1)|(npMask[:,:,4]==1),255,0) # one color
    finalnpMask = np.concatenate((finalnpMask,temp[:, :, None]),axis=2)
    # plt.imshow(finalnpMask)
    # plt.show()
    return finalnpMask

def evalutation(pred, target):
    # PQ
    pq_score = get_fast_pq(target, pred)[0]
    print("Detection Quality (DQ):", pq_score[0])
    print("Segmentation Quality (SQ):", pq_score[1])
    print("Panoptic Quality (PQ):", pq_score[2])
    dice_score = get_dice_1(target, pred)
    print("Dice score:", dice_score, "\n")
    return pq_score, dice_score

def test(model, device, data, target):
    print("Input Image")
    # plt.imshow(data[0].numpy().transpose(1, 2, 0))
    # plt.show()
    outputs = model(data.to(device))[0]
    pred = outputs.to('cpu').detach()
    pred=F.softmax(pred, dim=0)# along the channel
    pred=pred.numpy()

    print("Predicted Mask Sigmoid")
    pred[pred.max(axis=0,keepdims=1) == pred] = 1
    pred[pred.max(axis=0,keepdims=1) != pred] = 0
    # printColoredMask(pred)
    # print("Actual Mask")
    # printColoredMask(target[0].numpy())

    evalutation(pred, target)
    return pred


def crop_imageUI(input_path, save_path, img_name, label_name, image_size=1000, crop_size=500):
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
    dictmat = loadmat("./images/"+label_name)
    (top_left, top_right, bottom_left, bottom_right, center) = T.FiveCrop(size=(crop_size, crop_size))(orig_img)
    cropped = [top_left, top_right, bottom_left, bottom_right]
    if crop_size == 250:
        cropped = [top_left, top_right, bottom_left, bottom_right]
    save_name = img_name.split(".")[0]
    
    for img_idx in range(len(cropped)):
        name = ""
        labels = {}
        if img_idx == 0:                
            name = "top_left"
            if crop_size == 250:
                name = "0"
            labels["inst_map"] = dictmat["inst_map"][0:crop_size, 0:crop_size]
            labels["type_map"] = dictmat["type_map"][0:crop_size, 0:crop_size]
            labels["inst_type"] = dictmat["inst_type"][0:crop_size, 0:crop_size]
            labels["inst_centroid"] = dictmat["inst_centroid"][0:crop_size, 0:crop_size]
        elif img_idx == 1:
            name = "top_right"
            if crop_size == 250:
                name = "1"
            labels["inst_map"] = dictmat["inst_map"][0:crop_size, image_size-crop_size:image_size]
            labels["type_map"] = dictmat["type_map"][0:crop_size, image_size-crop_size:image_size]
            labels["inst_type"] = dictmat["inst_type"][0:crop_size, image_size-crop_size:image_size]
            labels["inst_centroid"] = dictmat["inst_centroid"][0:crop_size, image_size-crop_size:image_size]
        elif img_idx == 2:
            name = "bottom_left"
            if crop_size == 250:
                name = "2"
            labels["inst_map"] = dictmat["inst_map"][image_size-crop_size:image_size, 0:crop_size]
            labels["type_map"] = dictmat["type_map"][image_size-crop_size:image_size, 0:crop_size]
            labels["inst_type"] = dictmat["inst_type"][image_size-crop_size:image_size, 0:crop_size]
            labels["inst_centroid"] = dictmat["inst_centroid"][image_size-crop_size:image_size, 0:crop_size]
        elif img_idx == 3:
            name = "bottom_right"
            if crop_size == 250:
                name = "3"
            labels["inst_map"] = dictmat["inst_map"][image_size-crop_size:image_size, image_size-crop_size:image_size]
            labels["type_map"] = dictmat["type_map"][image_size-crop_size:image_size, image_size-crop_size:image_size]
            labels["inst_type"] = dictmat["inst_type"][image_size-crop_size:image_size, image_size-crop_size:image_size]
            labels["inst_centroid"] = dictmat["inst_centroid"][image_size-crop_size:image_size, image_size-crop_size:image_size]
        elif img_idx == 4:
            if crop_size == 500:
                name = "center"
        cropped[img_idx].save(save_path +"/Images/" + save_name + "_" +name+".png")
        savemat(save_path + "/Labels/" + save_name + "_" +name+".mat", labels)  
    return


def cropping(input_path, img_name, label_name):
    save_folder_500 = "crop_ui_500"
    save_path_500 = input_path + save_folder_500
    os.makedirs(save_path_500, exist_ok=True)
    os.makedirs(save_path_500+"/Images", exist_ok=True)
    os.makedirs(save_path_500+"/Labels", exist_ok=True)
    crop_imageUI(input_path, save_path_500, img_name, label_name, image_size=1000, crop_size=500)
    
    
    ## crop to 250x250 
    # fileNames = [i[:-4] for i in sorted(os.listdir(save_path_500))]
    # save_folder_250 = input_path+"crop_ui_250"
    # os.makedirs(save_folder_250, exist_ok=True)
    # for i in range(len(fileNames)):
    #     crop_imageUI(save_path_500+"/", save_folder_250, fileNames[i], image_size=500, crop_size=250)
    return