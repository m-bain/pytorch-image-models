import torch
from timm.models import vision_transformer
from PIL import Image
from torch import nn
import os
import tarfile
import numpy as np
import random
import io
import torch
from torchvision import transforms

STRIDE = 1
EXTRACTION_FPS = 25
NUM_FRAMES = 4
def _sample_video_idx(vlen):
    frame_stride = STRIDE * EXTRACTION_FPS
    target_frames = min((vlen // frame_stride) + 1, NUM_FRAMES)
    sample_range = (target_frames - 1) * frame_stride
    possible_start_idx = range(0, vlen - sample_range + 1)
    if len(possible_start_idx) == 0:
        print(vlen, sample_range)

    start_idx = random.choice(possible_start_idx)
    # center crop
    # start_idx = possible_start_idx[len(possible_start_idx) // 2]
    res = np.linspace(start=start_idx, stop=start_idx + sample_range - 1, num=target_frames,
                      endpoint=False).astype(int)
    return res

### instantiate model
model = vision_transformer.timesformer_base_patch16_224(num_frames=NUM_FRAMES)
model.head = nn.Identity()
model.pre_logits = nn.Identity()
model = model.cuda()
# load pretrained
checkpoint = torch.load(
    '/work/maxbain/Libs/Alignment/saved/models/msrvtt_jsfusion_baseline__timesformer_ccep1__distilbert_base_uncased__NormSoftMax/0305_160357/model_best.pth')
model.load_state_dict(checkpoint)  # , strict=False)

### get example video.
frame_dir = '/scratch/shared/beegfs/albanie/shared-datasets/MSRVTT/high-quality/frames-25fps-256px/tars/all'
target_video = 'video9991'
tar_fp = os.path.join(frame_dir, target_video + '.tar')

tf = tarfile.open(tar_fp)
contents = tf.getmembers()
target_names = np.array(sorted([x.name for x in contents if x.name.endswith('.jpg')]))
vid_len = len(target_names)
frame_idxs = _sample_video_idx(vid_len)
imgs = []

for fidx in frame_idxs:
    if fidx < len(target_names):
        fp = target_names[fidx]
        image = tf.extractfile(fp)
        image = image.read()
        image = Image.open(io.BytesIO(image))
        image = transforms.ToTensor()(image)
        imgs.append(image)

if imgs == []:
    imgs = [transforms.ToTensor()(Image.new('RGB', (300, 300), (255, 255, 255)))]

imgs = torch.stack(imgs)
input_res = 224
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
tsfm = transforms.Compose([
                transforms.CenterCrop(256),
                transforms.Resize(input_res),
                normalize
    ])

imgs = tsfm(imgs)

import pdb; pdb.set_trace()