import albumentations as albu

def get_training_augmentation():
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
        albu.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),
        albu.RandomCrop(height=320, width=320, always_apply=True),
        albu.GaussNoise(p=0.2),
        albu.Perspective(scale=(0.05, 0.1), p=0.5),
        albu.OneOf([
            albu.CLAHE(p=1),
            albu.RandomBrightnessContrast(brightness_limit=0, contrast_limit=(0.2, 0.5), p=1),
            albu.RandomGamma(p=1),
        ], p=0.9),
        albu.OneOf([
            albu.Sharpen(p=1),
            albu.Blur(blur_limit=3, p=1),
            albu.MotionBlur(blur_limit=3, p=1),
        ], p=0.9),
        albu.OneOf([
            albu.RandomBrightnessContrast(brightness_limit=0, contrast_limit=(0.2, 0.5), p=1),
            albu.HueSaturationValue(p=1),
        ], p=0.9),
    ]
    return albu.Compose(train_transform)

def get_validation_augmentation():
    test_transform = []
    return albu.Compose(test_transform)

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def get_preprocessing(preprocessing_fn):
    def preprocess(image):
        image = preprocessing_fn(image)
        image = to_tensor(image)
        return image
    return preprocess