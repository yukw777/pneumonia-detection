import argparse
import mrcnn.model as modellib

from imgaug import augmenters as iaa
from config import from_config_file
from dataset import get_train_and_val

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='path to train config file')
    parser.add_argument('dicom_dir', help='path to directory with train dicom files')
    parser.add_argument('label_file', help='path to label file')
    parser.add_argument('model_dir', help='directory for logs and trained models')
    parser.add_argument('--coco', help='path to pretrained coco weights')
    parser.add_argument('--checkpoint', help='path to checkpoint weights')
    parser.add_argument('--head-epochs', help='number of epochs for head training. Default 2', type=int, default=2)
    parser.add_argument('--all-epochs', help='list of number of epochs for all layer training. Default 6 16', nargs='+', type=int, default=[6, 16])
    args = parser.parse_args()

    config = from_config_file(args.config)
    model = modellib.MaskRCNN(mode='training', config=config, model_dir=args.model_dir)
    if args.coco:
        model.load_weights(args.coco, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask",
        ])
    elif args.checkpoint:
        model.load_weights(args.checkpoint)

    # Image augmentation
    augmentation = iaa.Fliplr(0.5)

    # train and validation sets
    dataset_train, dataset_val = get_train_and_val(args.dicom_dir, args.label_file)

    # Train Mask-RCNN Model
    import warnings
    warnings.filterwarnings("ignore")

    print('Train network heads')
    lr = config.LEARNING_RATE*2
    print('Learning rate %f' % lr)
    # Train the heads with higher learning rate first
    model.train(dataset_train, dataset_val,
                learning_rate=lr,
                epochs=args.head_epochs,
                layers='heads',
                augmentation=augmentation)

    print('Train all layers')
    lr = config.LEARNING_RATE
    for i, epoch in enumerate(args.all_epochs):
        print('Stage %d' % (i+1))
        lr /= 5
        print('Learning rate %f' % lr)
        model.train(dataset_train, dataset_val,
                learning_rate=lr,
                epochs=epoch,
                layers='all',
                augmentation=augmentation)
