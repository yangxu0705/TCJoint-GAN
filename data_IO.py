#sample = np.load('"D:\\GAN\\1111\\learned-models1\\3dGAN-tf2\\3dGAN\\3dGAN-results.pickle5.npy"')
#print(sample.shape)

import sys
import os

import scipy.ndimage as nd
import scipy.io as io
import numpy as np
import matplotlib.pyplot as plt
import skimage.measure as sk
import mayavi.mlab as mlab

from mpl_toolkits import mplot3d
from stl import mesh
import trimesh
try:
    #import trimesh
    from stl import mesh
except:
    pass
    print('All dependencies not loaded, some functionality may not work')

# LOCAL_PATH = '/home/meetshah1995/datasets/ModelNet/3DShapeNets/volumetric_data/'
#LOCAL_PATH = 'D:/GAN/3DShapeNetsCode/3DShapeNets/'
#SERVER_PATH = 'D:/GAN/3DShapeNetsCode/3DShapeNets/'

def getVF(path):
    raw_data = tuple(open(path, 'r'))
    header = raw_data[1].split()
    n_vertices = int(header[0])
    n_faces = int(header[1])
    vertices = np.asarray([map(float, raw_data[i+2].split()) for i in range(n_vertices)])
    faces = np.asarray([map(int, raw_data[i+2+n_vertices].split()) for i in range(n_faces)])
    return vertices, faces

def plotFromVF(vertices, faces):
    input_vec = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            input_vec.vectors[i][j] = vertices[f[j], :]
    figure = plt.figure()
    axes = mplot3d.Axes3D(figure)
    figure.add_axes(axes)

    axes.add_collection3d(mplot3d.art3d.Poly3DCollection(input_vec.vectors))
    # scale = input_vec.points.flatten(-1)
    scale = input_vec.points.flatten(order = "C")
    axes.auto_scale_xyz(scale, scale, scale)
    plt.show()
    input_vec.save('matlab_joint.stl')

def plotFromVoxels(voxels):
    z, x, y = voxels.nonzero()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, -z, zdir='z', c='red')
    plt.show()

def getVFByMarchingCubes(voxels, threshold=0.5):
    # v, f = sk.marching_cubes(voxels, level=threshold)
    # v, f,_,_ = sk.marching_cubes_lewiner(voxels, threshold)
    v, f = sk.marching_cubes_classic(voxels, threshold)
    return v, f        # the lower the level is, the more detail of voxels are captured; the higher, the less noise in the volumetric model

def plotMeshFromVoxels(voxels, threshold=0.5):
    v, f = getVFByMarchingCubes(voxels, threshold)
    plotFromVF(v, f)

def plotVoxelVisdom(voxels, visdom, title):
    v, f = getVFByMarchingCubes(voxels)
    visdom.mesh(X=v, Y=f, opts=dict(opacity=0.5, title=title))

def plotFromVertices(vertices):
    figure = plt.figure()
    axes = mplot3d.Axes3D(figure)
    axes.scatter(vertices.T[0, :], vertices.T[1, :], vertices.T[2, :])
    plt.show()

def getVolumeFromOFF(path, sideLen=32):
    mesh = trimesh.load(path)  # 读取三维模型
    # volume = trimesh.voxel.Voxel(mesh, 0.5).raw
    volume = mesh.voxelized(0.5)
    (x, y, z) = map(float, volume.shape)  # map()会根据提供的函数对指定序列做映射。
    volume = nd.zoom(volume.matrix.astype(float),
                     (sideLen/x, sideLen/y, sideLen/z),
                     order=1,
                     mode='nearest')
    volume[np.nonzero(volume)] = 1.0
    return volume.astype(np.bool)

def getVoxelFromMat(path, cube_len=64):
    voxels = io.loadmat(path)['instance']
    voxels = np.pad(voxels, (1, 1), 'constant', constant_values=(0, 0))  # 常常采用numpy.pad()进行填充操作
    #if cube_len != 32 and cube_len == 64:
        #voxels = nd.zoom(voxels, (2, 2, 2), mode='constant', order=0)
    return voxels

def getAll(obj='airplane', train=True, is_local=False, cube_len=64, obj_ratio=1.0):
    objPath = SERVER_PATH + obj + '/30/' # type: ignore
    if is_local:
        objPath = LOCAL_PATH + obj + '/30/' # type: ignore
    objPath += 'train/' if train else 'test/'
    fileList = [f for f in os.listdir(objPath) if f.endswith('.mat')]
    fileList = fileList[0:int(obj_ratio*len(fileList))]
    volumeBatch = np.asarray([getVoxelFromMat(objPath + f, cube_len) for f in fileList], dtype=np.bool)
    return volumeBatch

if __name__ == '__main__':
    #path = sys.argv[1]
    #path = os.path.abspath(sys.argv[1])   # sys.argv[]是用来获取命令行参数的
    '''path = '.\\3DShapeNets\\3D\\chair_0001.off' # path = '3DShapeNets\\3D\\chair.off'
    volume = getVolumeFromOFF(path)
    #plotFromVoxels(volume)
    plotMeshFromVoxels(volume, threshold=0.5)'''
    # 下面的代码是用来查看生成的东西的
    samples = np.load("E:\\PythonProject11\\learned-models\\3dGAN-tf2\\3dGAN\\3dGAN-results.pickle20.npy")
    print(samples.shape)
    chair = samples[2]
    chair = chair.reshape([64, 64, 64])
    print(chair.shape)
    #chair = (chair_0 + 1) / 2.0
    print(chair.min(), chair.max())
    chair = chair.round()  # round values to 0.0 or 1.0
    print(chair.min(), chair.max())
    plotMeshFromVoxels(chair, threshold=0.5)
    plotFromVoxels(chair)
    #设置阈值以确定哪些部分应该被可视化为实体
    threshold = 0.5
    voxels = (chair > threshold)

    # 获取体素数组中值为实体的索引
    xx, yy, zz = np.where(voxels)

    # 设置Mayavi场景，背景颜色为白色
    mlab.figure(bgcolor=(1, 1, 1))  # 设置背景颜色为白色

    # 将RGB值从0-255转换为0-1范围
    red = 255 / 255
    green = 147 / 255
    blue = 147 / 255

    # 用指定的RGB颜色和立方体图标可视化体素
    mlab.points3d(xx, yy, zz, mode="cube", color=(red, green, blue), scale_factor=1)  # 自定义颜色

    # 显示图形
    mlab.show()
    xx, yy, zz = np.where(chair == 1)
    mlab.points3d(xx, yy, zz,
                          mode="cube",
                          color=(0.9, 0.5, 0.5),
                          scale_factor=1)
    mlab.show()
    # # # # #
    


    #with open('test.mat', 'wb') as file:  # need 'wb' in Python3
    #    io.savemat(file, {'faces': f})
    #    io.savemat(file, {'vertices': v})

    #from scipy import io
    #mat = np.load('rlt_gene_features.npy-layer-3-train.npy')
    #io.savemat('gene_features.mat', {'gene_features': mat})