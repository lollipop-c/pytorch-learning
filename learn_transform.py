from torchvision import transforms
from PIL import Image


def trans(img_item_path):
    resize = 224
    tf = transforms.Compose([
        lambda x:Image.open(img_item_path),
        transforms.Resize((int(resize*1.25), int(resize*1.25))),
        transforms.RandomRotation(15),
        transforms.CenterCrop(resize),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.2, 0.2, 0.2])
        ])
    img = tf(img_item_path)
    return img
