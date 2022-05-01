# inference for deployment
# pre requisite
"""
pip install pytesseract

start server : uvicorn main:app --reload
python -m uvicorn --host 0.0.0.0 --port 8001 main:app

"""

import torch
import os
import pickle
import numpy as np
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from src.utils import *
from src.model import *
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import cv2
import traceback
import shutil
# import easyocr

# pytesseract.pytesseract.tesseract_cmd = r'C:\Users\USER\AppData\Local\Tesseract-OCR\tesseract.exe'
#create a folder an keep only model export with name as export.pkl
# model_path='E:/GoogleDrive/DR/efficientnet_implementation/paperspace_models'
model_path='models'
unet_model_name = 'galucoma_segmentation_v0.1.pth'
regression_model_name = 'glaucoma_classifier_v0.1.pkl'
results_dir = 'results'

output_size=(256,256)

# Device
device = try_gpu()

# Load model and classifier
model = UNet(n_channels=3, n_classes=2).to(device)
model.load_state_dict(torch.load(model_path+'/'+unet_model_name,map_location=device))
with open(model_path+'/'+regression_model_name, 'rb') as clf_file:
    clf = pickle.load(clf_file)
_=model.eval()

# ---------------------------------------    
def infer(image_path):
    ouput_file_name = results_dir +'/' +os.path.basename(image_path).split('.')[0]+'.png'

    img = np.array(Image.open(image_path).convert('RGB'))
    img = vertical_crop(img)
    img = extract_eye_horizontal_vertical_edge(img)
    img = transforms.functional.to_tensor(img)
    img = transforms.functional.resize(img, output_size, interpolation=Image.BILINEAR)

    logits = model(img[None,:])

    # Compute segmentation metric
    pred_od = refine_seg((logits[:,0,:,:]>=0.5).type(torch.int8).cpu()).to(device)
    pred_oc = refine_seg((logits[:,1,:,:]>=0.5).type(torch.int8).cpu()).to(device)

    pred_od = pred_od.cpu().numpy()
    pred_oc = pred_oc.cpu().numpy()

    # Compute and store vCDRs
    pred_vCDR = vertical_cup_to_disc_ratio(pred_od, pred_oc)
    print("vCDR:",pred_vCDR[0])
    test_classif_preds = clf.predict_proba(np.array(pred_vCDR).reshape(-1,1))[:,1]
    print("Classification : ",test_classif_preds[0])

    pred_seg = np.zeros((img.shape[1],img.shape[2]), dtype=np.uint8)
    pred_seg[:,:]=255
    pred_seg[pred_od[0]==1.0]=128
    pred_seg[pred_oc[0]==1.0]=0

    # ax4.imshow(Image.open(image_path).convert('RGB'), interpolation='nearest')
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3,figsize=(20,10))
    ax1.imshow(img.permute(1,2,0), interpolation='nearest')
    ax1.title.set_text("Input")
    ax1.grid(False)
    ax2.imshow(img.permute(1,2,0), interpolation='nearest')
    ax2.imshow(cv2.cvtColor(pred_seg, cv2.COLOR_BGR2RGB), alpha=0.4)
    ax2.title.set_text("With Segmentation on Input")
    ax2.grid(False)
    ax3.imshow(cv2.cvtColor(pred_seg, cv2.COLOR_BGR2RGB))
    ax3.title.set_text("OD and OC")
    ax3.grid(False)
    plt.savefig(ouput_file_name)
    #plt.show()

    if test_classif_preds[0] <= 0.5:
        response = "No apparent signs of Glaucoma detected."
    elif test_classif_preds[0] > 0.5:
        response = "Glaucoma detected."


    return str(pred_vCDR[0]), response

def check_if_valid_image_type(image_full_path):
     
    return imghdr.what('/tmp/bass')


# image_path = 'E:/DATASET/resized_aptos2019_eyepack2015/resized_train_19/00a8624548a9.jpg'
# print(infer(image_path))

app = FastAPI()

class GalucomaAnalyzer(BaseModel):
    model_response: str
    model_coef: str

@app.post("/galucoma_analyze", response_model=GalucomaAnalyzer)
async def analyze_route(unique_id: str,file: UploadFile = File(...)):

    pred = 'NA'
    response = 'NA'

    try:         
        image_full_path = 'images/' + str(unique_id) + '.jpg'
        if os.path.exists(image_full_path):
            raise Exception("File already present with given path:"+image_full_path)

        with open(image_full_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        img = cv2.imread(image_full_path)

        img_dimensions = img.shape
        print(img_dimensions)
        print(f'Input image width: ', img_dimensions[0])
        print(f'Input image height:' , img_dimensions[1])
        print(f'Input image channels: ', img_dimensions[2])
        
        if img_dimensions[0] == 720 and img_dimensions[1] == 1280 and img_dimensions[2] == 3:
            print('(720, 1280, 3)')

        img_dimensions = img.shape
        print(f'Image Dimension:', img_dimensions[0])

        pred, response = infer(image_full_path)

    except Exception as e:
        print("Exception : ",e)
        print("Stack Trace : ")
        traceback.print_exc()
    finally:
        return{
            'model_response': response,
            'model_coef': pred
        }