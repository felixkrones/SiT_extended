import os
import sys

from dataset_functions.Pets import pets
from dataset_functions.MIMIC import MIMIC_Dataset

def build_dataset(args, is_train, trnsfrm=None, training_mode='finetune'):


    if args.data_set == 'Pets':
        split = 'trainval' if is_train else 'test'
        dataset = pets(os.path.join(args.data_location, 'Pets_dataset'), split=split, transform=trnsfrm)
        
        nb_classes = 37

    elif args.data_set == 'MIMIC':
        print(f"Getting MIMIC data, downsampled to {args.n_samples} samples with {args.input_channels} channels, {args.large_img} using large images.")
        dataset = MIMIC_Dataset(datapath=args.data_location, transform=trnsfrm, split="train", input_channels=args.input_channels, n_samples = args.n_samples, large_img=args.large_img)
        nb_classes = dataset.num_classes

    else:
        print('dataloader of {} is not implemented .. please add the dataloader under datasets folder.'.format(args.data_set))
        sys.exit(1)
        
    return dataset, nb_classes
