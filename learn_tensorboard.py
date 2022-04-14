from torch.utils.tensorboard import SummaryWriter
import cv2


def test_add_scalar():
    writer = SummaryWriter("logs")
    for i in range(100):
        writer.add_scalar("y=2x", 2 * i, i)
    writer.close()


def test_add_image(img, label, step):
    writer = SummaryWriter("logs")
    for i in range(100):
        writer.add_image("{0}".format(label), img, step, dataformats="HWC")
    writer.close()
