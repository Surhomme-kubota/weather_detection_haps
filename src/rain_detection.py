
import logging
from tqdm import tqdm
import pandas as pd
from typing import Tuple
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import EvalDataset
from efficient import effnetv2_l
from preactresnet import PreActResNet34

from config.config import TRAIN_CONDITIONS, MODEL_SETTINGS, MODEL_SAVE_PATHS_STR 


# ログの設定
def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filename = str(Path(__file__).resolve().parent.parent / 'results' / 'logs' / 'evaluation.log'),
        filemode='w'
    )
    logger = logging.getLogger('evaluation_logger')
    return logger

# モデルのインスタンスとローディング
def load_models(device: str) -> Tuple[torch.nn.Module, torch.nn.Module]:
    model1 = PreActResNet34(n_c=3, num_classes=MODEL_SETTINGS['num_classes']).to(device)
    model2 = effnetv2_l(num_classes=MODEL_SETTINGS['num_classes'], width_mult=1.0).to(device)
    model1.load_state_dict(torch.load(MODEL_SAVE_PATHS_STR['model1']))
    model2.load_state_dict(torch.load(MODEL_SAVE_PATHS_STR['model2']))
    return model1, model2

# Stacking Model のクラス定義
class StackingModel(torch.nn.Module):
    def __init__(self,
                 model1: torch.nn.Module,
                 model2: torch.nn.Module,
                 num_classes: int) -> None:
        
        super(StackingModel, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.fc = torch.nn.Linear(model1.fc.out_features + model2.classifier.out_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.model1(x)
        x2 = self.model2(x)
        x = torch.cat((x1, x2), dim=1)
        x = self.fc(x)
        return x

# モデルの評価を行う関数
def evaluate_models(device: str,
                    stacked_model: torch.nn.Module,
                    test_loader: DataLoader, save_file = None) -> None:
    
    stacked_model.eval()
    labels_language = ['晴れまたは曇り', '雨']
    judgements = []
    names = []
    with torch.no_grad():
        for img, _, name in tqdm(test_loader):
            img = img.to(device)
            b, ncrops, c, h, w = img.size()
            output = stacked_model(img.view(-1, c, h, w))
            output = output.view(b, ncrops, -1).mean(1)
            predicted_labels = output.argmax(1)
            pred = predicted_labels.tolist()[0]
            judgement = labels_language[pred]
            judgements.append(judgement)
            names.append(name[0].split('/')[-1])

    # DataFrame を作成
    df = pd.DataFrame({
        "file": names,
        'Judgement': judgements
    })

    # CSV ファイルに保存
    if save_file:
        save_path = Path(__file__).resolve().parent.parent / 'results' / 'pred_result' / 'judgements.csv'
        df.to_csv(save_path, index=False)
        
    return judgements

def rain_main() -> None:
    logger = setup_logging()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    test_transform = transforms.Compose([
        transforms.FiveCrop(224),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops]))
    ])
    test_dataset = EvalDataset(transforms=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=TRAIN_CONDITIONS['evaluation_batch_size'], shuffle=False)

    model1, model2 = load_models(device)
    stacked_model = StackingModel(model1, model2, MODEL_SETTINGS['num_classes']).to(device)
    stacked_model.load_state_dict(torch.load(MODEL_SAVE_PATHS_STR['default_model']))

    rain_detection = evaluate_models(device, stacked_model, test_loader)
    return rain_detection