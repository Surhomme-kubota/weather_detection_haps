import torch
from torch.utils.data import DataLoader
import os
import numpy as np
import cv2

from cloud_dataset import TestDataset
from cloud_transform import get_validation_augmentation, get_preprocessing
from unet import SwishUNet
from config.white_cloud_config import ENCODER, CLASSES, DEVICE, TEST_FILE_FOLDER_PATH, OUTPUT_FILE_FOLDER_PATH, MODEL_PATH
from tqdm import tqdm


def load_model(model_path):
    model = SwishUNet(encoder_name=ENCODER, num_classes=len(CLASSES)).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model


def save_predicted_masks(model, data_loader, output_dir, filenames):
    with torch.no_grad():
        for idx, image in enumerate(tqdm(data_loader)):
            image = image.to(DEVICE)
            output = model(image)
            output = torch.nn.functional.interpolate(output, size=image.shape[2:], mode='bilinear', align_corners=False)
            
            # Save predicted masks
            output = output.cpu().numpy()
            for i in range(output.shape[0]):
                pred_mask = (output[i, 0] > 0.5).astype(np.uint8) * 255
                image_name = f"pred_mask_{idx * data_loader.batch_size + i}.png"
                cv2.imwrite(os.path.join(output_dir, image_name), pred_mask)


def white_cloud_detection_main():
    
    # File directory
    test_file_folder_path = TEST_FILE_FOLDER_PATH
    filename = os.listdir(TEST_FILE_FOLDER_PATH)
    if len(filename) != 1:
        raise ValueError("画像ファイルが複数枚エントリーされています")
    test_file_path = str(test_file_folder_path / filename[0])   
    
    # Output file directory
    output_dir = str(OUTPUT_FILE_FOLDER_PATH)
    os.makedirs(output_dir, exist_ok=True)
    
    print("解析する画像ファイル名:", filename)
    
    preprocessing_fn = lambda x: x
    
    test_dataset = TestDataset(
        test_file_path,
        classes=CLASSES,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
    )

    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)
    
    model_path = str(MODEL_PATH)
    model = load_model(model_path)
    
    save_predicted_masks(model, test_loader, output_dir, filename)
