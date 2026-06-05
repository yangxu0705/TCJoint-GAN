import numpy as np
import sys
import os
import scipy.io
from path import Path
from tqdm import tqdm

if sys.argv[-1] == '-v': # this will allow you to visualize the models as they are made, more of a sanity check 
    import mayavi.mlab
    import matplotlib.pyplot as plt
    from scipy import ndimage
    from mpl_toolkits.mplot3d import Axes3D
# os即operating system（操作系统），Python 的 os 模块封装了常见的文件和目录操作。
# os.path模块主要用于文件的属性获取,exists是“存在”的意思，所以顾名思义，os.path.exists()就是判断括号里的文件是否存在的意思，括号内的可以是文件路径。

# tqdm是Python中专门用于进度条美化的模块，通过在非while的循环体内嵌入tqdm，可以得到一个能更好展现程序运行过程的提示进度条
instances = {}
class_id_to_name = {  # 定义标签字典，每一个数字所代表的图像类别的名称
    "1": "bathtub",
    "2": "bed",
    "3": "chair",
    "4": "desk",
    "5": "dresser",
    "6": "monitor",
    "7": "night_stand",
    "8": "sofa",
    "9": "table",
    "10": "toilet"
}
class_name_to_id = { v : k for k, v in class_id_to_name.items() }
class_names = set(class_id_to_name.values())

if not os.path.exists('data/train/'):  # 判断一个目录是否存在
    os.makedirs('data/train/')  # os.makedirs(path) 多层创建目录  os.mkdir(path) 创建目录
if not os.path.exists('data/test/'):
    os.makedirs('data/test/')

base_dir = Path('E:\\PythonProject11\\3DShapeNets\\volumetric_data\\chair\\30\\train').expand()   #将路径形式的字符串转为路径，然后清理文件名

for fname in tqdm(sorted(base_dir.walkfiles('*.mat'))):
    if fname.endswith('test_feature.mat') or fname.endswith('train_feature.mat'):  # 判断字符串是否以指定字符或子字符串结尾，常用于判断文件类型
        continue
    elts = fname.splitall()
    info = Path(elts[-1]).stripext().split('_')
    if len(info)<3: continue
    if info[0] == 'discriminative' or info[0] == 'generative' : continue 
    instance = info[1]
    rot = int(info[2])
    split = elts[-2]
    classname = elts[-4].strip()
    if classname in class_names:
        dest = 'data/'+split+'/' + classname + '/'
        if not os.path.exists(dest):
            os.makedirs(dest)
        arr = scipy.io.loadmat(fname)['instance'].astype(np.uint8)
        matrix = np.zeros((64,)*3, dtype=np.uint8)
      
        matrix[:, :, :] = arr

        # matrix[1:-1, 1:-1, 1:-1] = arr
        if sys.argv[-1] == '-v':
            xx, yy, zz = np.where(matrix >= 0.3)
            mayavi.mlab.points3d(xx, yy, zz,
                                 color=(1, 0, 1),
                                 scale_factor=1)

            mayavi.mlab.show()
        # saves the models by instance name, and then rotation 
        np.save(dest +  instance + '_' + str(rot) , matrix)
        


