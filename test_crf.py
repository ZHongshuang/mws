import os
import numpy as np
import matplotlib.pyplot as plt
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral
import PIL.Image as Image
import multiprocessing
import pdb

img_root = '/home/zhang/data/datasets/saliency_Dataset/DUTS/DUT-train/images'
prob_root ='/home/zhang/mws2020/data/share-data/DUT-train-sal0'
output_root = '/home/zhang/mws2020/data/share-data/DUT-train-sal0_msk_crf_bin'


if not os.path.exists(output_root):
    os.mkdir(output_root)

files = os.listdir(prob_root)
# for img_name in files:
#for i, img_name in enumerate(files):
def myfunc(img_name):
    img = Image.open(os.path.join(img_root, img_name[:-4]+'.jpg'))
    img = np.array(img, dtype=np.uint8)
    if len(img.shape) < 3:
        img = np.stack((img, img, img), 2)
    H, W, _ = img.shape
    probs = Image.open(os.path.join(prob_root, img_name[:-4]+'.png'))
    probs = probs.resize((W, H))
    probs = np.array(probs)
    probs = probs.astype(np.float)/255.0
    probs = np.concatenate((1-probs[None, ...], probs[None, ...]), 0)
    # Example using the DenseCRF class and the util functions
    d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], 2)
    # get unary potentials (neg log probability)
    U = unary_from_softmax(probs)
    d.setUnaryEnergy(U)
    # This creates the color-dependent features and then add them to the CRF
    feats = create_pairwise_bilateral(sdims=(80, 80), schan=(13, 13, 13),
                                      img=img, chdim=2)
    d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
    d.addPairwiseEnergy(feats, compat=10,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)
    # Run five inference steps.
    Q = d.inference(5)
    # Find out the most probable class for each pixel.
    MAP = np.argmax(Q, axis=0).reshape((H, W))
    # bb = np.array(Q)[0].reshape(img.shape[:2])
    MAP = (MAP*255).astype(np.uint8)
    img = np.concatenate((img, MAP[..., None]), 2)
    msk = Image.fromarray(img)
    msk.save(os.path.join(output_root, img_name), 'png')

if __name__ == '__main__':
    print('start crf')
    pool = multiprocessing.Pool(processes=8)
    pool.map(myfunc, files)
    pool.close()
    pool.join()
    print('done')
