from torchvision import datasets

ds = datasets.UCF101(root='/datasets/UCF101-24',
                     annotation_path='/datasets/UCF101-24/UCF101_24Action_Detection_Annotations/UCF101_24Action_Detection_Annotations',
                     frames_per_clip=4)

import pdb; pdb.set_trace()