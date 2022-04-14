import read_data
import learn_tensorboard
import learn_transform

if __name__ == '__main__':
    root_dir = "dataset/train"
    ants_label_dir = "ants_image"
    bees_label_dir = "bees_image"
    ants_dataset = read_data.MyData(root_dir, ants_label_dir)
    bees_dataset = read_data.MyData(root_dir, bees_label_dir)
    train_dataset = bees_dataset + ants_dataset

    # img, label = train_dataset[0]
    # img.show()
    step = 0
    for img, label in train_dataset:
        print(step)
        img = learn_transform.trans(img)
        learn_tensorboard.test_add_image(img, label, step)
        step = step+1
