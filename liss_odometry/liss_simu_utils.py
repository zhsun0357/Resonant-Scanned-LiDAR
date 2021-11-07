import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
import scipy.io
import pickle

from scipy.spatial import KDTree as kdtree
import sys
sys.setrecursionlimit(10000)

class Liss_Simu_Utils():
    def __init__(self, calib_path):
        self.calib = self.read_calib_file(calib_path)
        self.velo_to_cam = np.zeros((4,4))
        self.velo_to_cam[:3,:] = self.calib['Tr'].copy().reshape(3,4)
        self.velo_to_cam[3,:3] = 0.0
        self.velo_to_cam[3,3] = 1.0
        self.P_rect = self.calib['P2'].copy().reshape(3,4)

        print(self.velo_to_cam)
        print(self.P_rect)

    """
    coordinate transform helpers for KITTI
    """
    def card2hom(self, pc):
        ## pc: Nxd, out: Nx(d+1)
        return np.hstack((pc, np.ones([pc.shape[0],1])))

    def velo2img(self, velo_pc, H, W):
        """
        input: 3d velo pc, Nx3 (z,x,y), crop range (image size in pixel)
        output: 2d image projection onto rectified camera coord, Nx2 (u,v)
        """
        velo_pc = self.card2hom(velo_pc).T ## 4xN
        # cam_pc = np.matmul(np.matmul(P_rect, velo_to_cam), velo_pc) ## 3xN
        cam_pc = np.matmul(self.P_rect, np.matmul(self.velo_to_cam, velo_pc))
        proj_pc = cam_pc[:2,:].copy().T ## Nx2
        proj_pc = proj_pc / np.expand_dims(cam_pc[2,:].copy(), axis = 1)

        mask = (proj_pc[:,1] >= 0) * (proj_pc[:,1] <= (H - 1)) * (proj_pc[:,0] >= 0) * (proj_pc[:,0] <= (W - 1))
        proj_pc = proj_pc[mask]
        cam_pc = cam_pc[:,mask]

        return proj_pc, cam_pc.T

    def velo2dmap(self, velo_pc, H, W):
        """
        input: velodyne point cloud, image height, width
        output: sparse depth map for inpainting
        """
        proj_pc, cam_pc  = self.velo2img(velo_pc)
        mask = (proj_pc[:,1] >= 0) * (proj_pc[:,1] <= (H - 1)) * (proj_pc[:,0] >= 0) * (proj_pc[:,0] <= (W - 1))
        proj_pc = proj_pc[mask]
        cam_pc = cam_pc[mask]

        dmap = np.zeros((H, W))
        count = np.zeros((H, W))
        for ii in np.arange(proj_pc.shape[0]):
            ind_x = int(np.floor(proj_pc[ii, 1]))
            ind_y = int(np.floor(proj_pc[ii, 0]))
            dmap[ind_x, ind_y] += cam_pc[ii,2]
            count[ind_x, ind_y] += 1

        dmap = np.divide(dmap, count, np.zeros((H, W)), where = count > 0)
        return dmap

    """
    lidar point cloud simulation
    """
    def solve_xyz_uvd(self, u,v,d):
        fx = self.P_rect[0,0]
        fy = self.P_rect[1,1]
        cx = self.P_rect[0,2]
        cy = self.P_rect[1,2]
        bx = -self.P_rect[0,3] / fx ## by = 0.0 since velodyne and camera are at same height

        x_cam = (u - cx) * d / fx + bx
        y_cam = (v - cy) * d / fy
        pc_cam = np.vstack((x_cam, y_cam, d, np.ones(x_cam.shape[0]))) ##Nx3
        pc_back = np.matmul(np.linalg.inv(self.velo_to_cam), pc_cam)

        return pc_back.T

    def solve_xyz_naive(self, lidar_raw):
        ind_lidar = np.argwhere(lidar_raw > 1.0) ## here we use 1m as the smallest z threshold
        v = ind_lidar[:,0].astype(np.float32)
        u = ind_lidar[:,1].astype(np.float32)
        d = lidar_raw[v.astype(np.int), u.astype(np.int)]

        pc_back = self.solve_xyz_uvd(u, v, d)
        return pc_back, ind_lidar


    """
    use KDTree for point cloud resampling
    """
    def get_lidar_data(self, dmap, theta, phi):

        H, W = dmap.shape
        fx = self.P_rect[0,0]
        fy = self.P_rect[1,1]
        cx = self.P_rect[0,2]
        cy = self.P_rect[1,2]
        bx = -self.P_rect[0,3] / fx ## by = 0.0 since velodyne and camera are at same height

        pc_back, _ = self.solve_xyz_naive(dmap) ## Nx3, (z, x, y)
        x_pc = pc_back[:,1]
        y_pc = pc_back[:,2]
        z_pc = pc_back[:,0]

        theta_pc = np.arctan2(np.sqrt(x_pc**2 + z_pc**2), y_pc)
        phi_pc = np.arctan2(x_pc, z_pc)

        tree = kdtree(np.vstack((theta_pc, phi_pc)).T)
        dis, idx = tree.query(np.vstack((theta, phi)).T)

        x_pc_scan = x_pc[idx]
        y_pc_scan = y_pc[idx]
        z_pc_scan = z_pc[idx]

        pc = np.vstack((np.asarray(x_pc_scan), np.asarray(y_pc_scan), np.asarray(z_pc_scan)))

        return pc

    def get_lidar_data_noedge(self, dmap, theta, phi):
        
        H = 375
        W = 1242
        pc0 = self.get_lidar_data(dmap, theta, phi).T
        pc = np.vstack((pc0[:,2], pc0[:,0], pc0[:,1])).T
        # pdb.set_trace()

        pc_img, cam_pc = self.velo2img(pc, H, W)
        pc_img_mask = (pc_img[:,0] > 1) * (pc_img[:,1] > 1) * (pc_img[:,0] < W-26) * (pc_img[:,1] < H-26)
        pc_img = pc_img[pc_img_mask]
        cam_pc = cam_pc[pc_img_mask]
        pc = self.solve_xyz_uvd(pc_img[:,0], pc_img[:,1], cam_pc[:,2])

        return pc

    """
    load calibration data helper functions, adapted from Frustum_Pointnet
    https://github.com/charlesq34/frustum-pointnets
    """
    def read_calib_file(self, filepath):
        ''' Read in a calibration file and parse into a dictionary.
        Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
        '''
        data = {}
        with open(filepath, 'r') as f:
            for line in f.readlines():
                line = line.rstrip()
                if len(line)==0: continue
                key, value = line.split(':', 1)
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass

        return data
