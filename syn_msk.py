import os
from PIL import Image
import numpy as np
import scipy
import scipy.stats
import random
import ipdb

obj_dir = '/home/zeng/data/datasets/obj_mr_msk_crf'
bg_dir = '/home/zeng/data/datasets/bg'
output_dir_img = '/home/zeng/data/datasets/syn_seg_mr/images'
output_dir_msk = '/home/zeng/data/datasets/syn_seg_mr/masks'


if not os.path.exists(output_dir_img):
    os.mkdir(output_dir_img)
if not os.path.exists(output_dir_msk):
    os.mkdir(output_dir_msk)

obj_names = os.listdir(obj_dir)
bg_names = os.listdir(bg_dir)

palette = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128,
           128, 128, 128, 64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128,
           64, 128, 128, 192, 128, 128, 0, 64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0, 0, 64, 128]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


# for obj_name, bg_name in zip(obj_names, bg_names):
def proc_one(bg_name, num):
    #ii = np.random.normal(loc=2, scale=1)
    ii = 1
    ii = 1 if ii <=0 else ii
    bg = Image.open(os.path.join(bg_dir, bg_name))
    sbc, sbr = bg.size
    ratio = 400.0 / max(sbr, sbc)
    bg = bg.resize((int(sbc * ratio), int(sbr * ratio)))
    bg = np.array(bg, dtype=np.uint8)
    r, c, _ = bg.shape
    mask = np.zeros((bg.shape[0], bg.shape[1], 1), dtype=np.uint8)
    locs = np.linspace(0, 1, ii+2)[1:-1]
    for i in range(ii):
        obj_name = random.choice(obj_names)
        obj = Image.open(os.path.join(obj_dir, obj_name))
        r_location = scipy.stats.norm.rvs(locs[i]-0.25, 0.2, size=1)[0] * r
        r_location = int(r_location)
        r_location = max(0, r_location)
        r_location = min(r_location, r - 1)

        c_location = scipy.stats.norm.rvs(locs[i]-0.25, 0.2, size=1)[0] * c
        c_location = int(c_location)
        c_location = max(0, c_location)
        c_location = min(c_location, c - 1)
        length = scipy.stats.norm.rvs(0.5, 0.07, size=1)[0] * max(r, c)
        length = max(length, 10)

        sbc, sbr = obj.size
        ratio = length / max(sbr, sbc)
        obj = obj.resize((int(sbc * ratio), int(sbr * ratio)))
        sbc, sbr = obj.size

        r_location_end = min(r_location + sbr, r)
        c_location_end = min(c_location + sbc, c)

        obj_r_end = min(r_location_end - r_location, sbr)
        obj_c_end = min(c_location_end - c_location, sbc)

        obj = np.array(obj, dtype=np.uint8)
        m_obj = obj[:, :, 3]
        m_obj[m_obj != 0] = 1
        m_obj = np.expand_dims(m_obj, 2)
        obj = obj[:, :, :3]

        bg[r_location:r_location_end, c_location:c_location_end] = \
            bg[r_location:r_location_end, c_location:c_location_end] * (1 - m_obj[:obj_r_end, :obj_c_end]) \
            + obj[:obj_r_end, :obj_c_end] * m_obj[:obj_r_end, :obj_c_end]
        mask[r_location:r_location_end, c_location:c_location_end] = \
            mask[r_location:r_location_end, c_location:c_location_end] * (1-m_obj[:obj_r_end, :obj_c_end]) \
            + (i+1)*m_obj[:obj_r_end, :obj_c_end]
    bg = Image.fromarray(bg)
    bg.save(os.path.join(output_dir_img, '{}_{}.jpg'.format( bg_name[:-4], num)))
    mask = mask[:, :, 0]
    mask = Image.fromarray(mask)
    mask = mask.convert('P')
    mask.putpalette(palette)
    mask.save(os.path.join(output_dir_msk, '{}_{}.png'.format( bg_name[:-4], num)))


if __name__ == "__main__":
    for i in range(3000):
        bg_name = random.choice(bg_names)
        proc_one(bg_name, i)
