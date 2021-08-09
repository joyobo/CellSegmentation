import numpy as np

def unpad(x, pad_width):
    slices = []
    for c in pad_width:
        e = None if c[1] == 0 else -c[1]
        slices.append(slice(c[0], e))
    return x[tuple(slices)]


def stitch_500(pred_dict):
    '''
    pred_dict: keys is filename, value is prediction
    '''
    if len(pred_dict) > 4:
        return "not right"
    pad_width = ((0, 0), (6, 6,), (6, 6)) # for size=500
    unpad_dict = {key: unpad(values, pad_width) for key, values in pred_dict.items()}
    stitch = np.zeros((5, 1000, 1000), dtype=int)
    for name, img in unpad_dict.items():
        size = img.shape[1]
        if "top_left" in name:
            stitch[:, 0:size, 0:size] = img   
        elif "top_right" in name:
            stitch[:, 0:size, 1000-size:1000] = img
        elif "bottom_left" in name:
            stitch[:, 1000-size:1000, 0:size] = img                          
        elif "bottom_right" in name:
            stitch[:, 1000-size:1000, 1000-size:1000] = img 
    return stitch

def stitch_250(pred_dict):
    '''
    pred_dict: keys is filename, value is prediction
    '''
    if len(pred_dict) < 16:
        print("not enough")
        return 
    
    pad_width = ((0, 0), (3, 3,), (3, 3)) # for size=250
    unpad_dict = {key: unpad(values, pad_width) for key, values in pred_dict.items()}
    stitch = np.zeros((5, 1000, 1000), dtype=int)
    for name, img in unpad_dict.items():
        size = img.shape[1]
        temp = np.zeros((5, 500, 500), dtype=int)
        if "top_left" in name:
            if "0" == name.split("_")[-1]:
                temp[:, 0:size, 0:size] = img  
            elif "1" == name.split("_")[-1]:
                temp[:, 0:size, 500-size:500] = img
            elif "2" == name.split("_")[-1]:
                temp[:, 500-size:500, 0:size] = img 
            elif "3"  == name.split("_")[-1]:
                temp[:, 500-size:500, 500-size:500] = img 
            stitch[:, 0:500, 0:500] = temp 
        elif "top_right" in name:
            if "0" == name.split("_")[-1]:
                temp[:, 0:size, 0:size] = img  
            elif "1" == name.split("_")[-1]:
                temp[:, 0:size, 500-size:500] = img
            elif "2" == name.split("_")[-1]:
                temp[:, 500-size:500, 0:size] = img 
            elif "3"  == name.split("_")[-1]:
                temp[:, 500-size:500, 500-size:500] = img 
            stitch[:, 0:500, 500:1000] = temp 
        elif "bottom_left" in name:
            if "0" == name.split("_")[-1]:
                temp[:, 0:size, 0:size] = img  
            elif "1" == name.split("_")[-1]:
                temp[:, 0:size, 500-size:500] = img
            elif "2" == name.split("_")[-1]:
                temp[:, 500-size:500, 0:size] = img 
            elif "3"  == name.split("_")[-1]:
                temp[:, 500-size:500, 500-size:500] = img 
            stitch[:, 500:1000, 0:500] = temp                             
        elif "bottom_right" in name:
            if "0" == name.split("_")[-1]:
                temp[:, 0:size, 0:size] = img  
            elif "1" == name.split("_")[-1]:
                temp[:, 0:size, 500-size:500] = img
            elif "2" == name.split("_")[-1]:
                temp[:, 500-size:500, 0:size] = img 
            elif "3"  == name.split("_")[-1]:
                temp[:, 500-size:500, 500-size:500] = img 
            stitch[:, 500:1000, 500:1000] = temp  
    return stitch