from torch.utils.tensorboard import SummaryWriter
import cv2


def test_add_scalar():
    writer = SummaryWriter("logs")
    for i in range(100):
        writer.add_scalar("y=2x", 2 * i, i)
    writer.close()


def test_add_image(img, label, step):
    writer = SummaryWriter("logs")
    writer.add_image(label, img, step)
    writer.close()
