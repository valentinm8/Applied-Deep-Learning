# Imports
import zipfile
import sys
import warnings
import pandas as pd
import numpy as np
import torch
import timm
from tabulate import tabulate
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import glob
import cv2
import matplotlib.pyplot as plt
import joblib
import gc
from glob import glob
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import albumentations as A
import os
warnings.filterwarnings("ignore")
print(np.__version__)
print(pd.__version__)
print(torch.__version__)
print(timm.__version__)

device = "cuda" if torch.cuda.is_available() else "cpu"

class Config:
    model_dir = "exp55"
    output_dir = "exp55"
    model_name = "beit_large_patch16_224"
    im_size = 224
    model_path = model_name
    base_dir = "input/petfinder-pawpularity-score"
    data_dir = base_dir
    img_test_dir = os.path.join(data_dir, "test")
    batch_size = 16

class PetDataset(Dataset):
    def __init__(self, image_filepaths, targets, transform=None):
        self.image_filepaths = image_filepaths
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.image_filepaths)

    def __getitem__(self, idx):
        image_filepath = self.image_filepaths[idx]
        with open(image_filepath, 'rb') as f:
            image = Image.open(f)
            image_rgb = image.convert('RGB')
        image = np.array(image_rgb)

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        image = image / 255
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        target = self.targets[idx]

        image = torch.tensor(image, dtype=torch.float)
        target = torch.tensor(target, dtype=torch.float)
        return image, target

class PetNet(nn.Module):
    def __init__(
            self,
            model_name=Config.model_path,
            out_features=1,
            inp_channels=3,
            pretrained=False,
    ):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=False, in_chans=3, num_classes=1)

    def forward(self, image):
        output = self.model(image)
        return output

def unzip(filename):
    with zipfile.ZipFile("static/files/uploads/"+filename, "r") as zip_ref:
        zip_ref.extractall("static/files/output")
    return 0

def load_files(foldername, file_names):
    train = pd.read_csv('input/petfinderdata/train-folds-1.csv')
    train['path'] = train['Id'].map(lambda x: 'input/petfinder-pawpularity-score/train/' + x + '.jpg')

    test = pd.DataFrame()
    test["Id"] = file_names
    #print(tabulate(test, headers='keys', tablefmt='fancy_grid'))
    # add new test set
    test['path'] = test['Id'].map(lambda x: 'static/files/output/' + foldername.split(".")[0] + "/" + x)
    test['app_path'] = test['Id'].map(lambda x: 'files/output/' + foldername.split(".")[0] + "/" + x)
    print(train.shape, test.shape)

    return train, test

def get_inference_fixed_transforms(mode=0, dim=224):
    if mode == 0:  # do not original aspects, colors and angles
        return A.Compose([
            A.SmallestMaxSize(max_size=dim, p=1.0),
            A.CenterCrop(height=dim, width=dim, p=1.0),
        ], p=1.0)
    elif mode == 1:
        return A.Compose([
            A.SmallestMaxSize(max_size=dim + 16, p=1.0),
            A.CenterCrop(height=dim, width=dim, p=1.0),
            A.HorizontalFlip(p=1.0)
        ], p=1.0)

def tta_fn(filepaths, model, ttas=[0, 1]):

    print('Image Size:', Config.im_size)
    model.eval()
    tta_preds = []
    for tta_mode in ttas:  #range(Config.tta_times):
        print(f'tta mode:{tta_mode}')
        test_dataset = PetDataset(
            image_filepaths=filepaths,
            targets=np.zeros(len(filepaths)),
            transform=get_inference_fixed_transforms(tta_mode, dim=Config.im_size)
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        #stream = tqdm(test_loader)
        tta_pred = []
        for images, target in test_loader:  #enumerate(stream, start = 1):
            images = images.to(device, non_blocking=True).float()
            target = target.to(device, non_blocking=True).float().view(-1, 1)
            with torch.no_grad():
                output = model(images)

            pred = (torch.sigmoid(output).detach().cpu().numpy() * 100).ravel().tolist()
            tta_pred.extend(pred)
        tta_preds.append(np.array(tta_pred))

    fold_preds = tta_preds[0]
    for n in range(1, len(tta_preds)):
        fold_preds += tta_preds[n]
    fold_preds /= len(tta_preds)

    del test_loader, test_dataset
    gc.collect()
    torch.cuda.empty_cache()
    return fold_preds

def run_model(test):
    filepaths = test['path'].values.copy()

    test_preds_model = []

    model_path = "input/petfinder-exp55/beit_large_patch16_224.pth"
    print(f'inference: {model_path}')
    model = PetNet(
        model_name=Config.model_path,
        out_features=1,
        inp_channels=3,
        pretrained=False
    )
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model = model.to(device)
    model = model.float()
    model.eval()
    test_preds_fold = tta_fn(filepaths, model, [0])
    test_preds_model.append(test_preds_fold)

    final_predictions55 = np.mean(np.array(test_preds_model), axis=0)
    #print(final_predictions55)
    test['Pawpularity'] = final_predictions55
    return test

def create_result_dict(test):

    test["Pawpularity"] = test["Pawpularity"].map('{:,.2f}'.format)
    paw_filename_dict_unsorted = dict(zip(test.app_path, test.Pawpularity))
    # sort dictionary
    paw_filename_dict_sorted = sorted(paw_filename_dict_unsorted.items(), key=lambda x: x[1], reverse=True)
    paw_filename_dict = dict(paw_filename_dict_sorted)

    return paw_filename_dict

def run_inference_fuc(foldername):

    # Unzip received file
    unzip(foldername)

    print(foldername)
    file_names = next(os.walk("static/files/output/" + foldername.split(".")[0]))[2]

    # Load files
    train, test = load_files(foldername, file_names)
    test = run_model(test)

    # display result in terminal
    print(tabulate(test, headers='keys', tablefmt='fancy_grid'))

    paw_filename_dict = create_result_dict(test)
    return paw_filename_dict

