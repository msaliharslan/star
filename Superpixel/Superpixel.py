#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 10:01:05 2021

@author: kutalmisince
"""

import numpy as np
import matplotlib.pyplot as plt

class Superpixel:

    def __init__(self, compactness = 8.0, tiling = 'iSQUARE', exp_area = 256.0, num_req_sps = 0, spectral_cost = 'Bayesian', spatial_cost = 'Bayesian'):
        
        '''
        compactness: weight of spatial distance, can be any floating number
        tiling: intial tiling {'RECT', 'HEX', 'IRECT'}
        exp_area: required average area of SPs (not used if num_req_sps is set)
        num_req_sps: number of required SPs
        spectral_cost: spectral cost function {'L2', 'Bayesian'}
        spatial_cost: spatial cost function {'L2', 'Bayesian'}
        '''
        self.compactness = float(compactness)
        self.tiling      = tiling  
        self.exp_area    = float(exp_area)
        self.num_req_sps = num_req_sps
        
        self.spectral_cost = spectral_cost
        self.spatial_cost = spatial_cost
        
        # hidden hyper-parameters
        self.measurement_precision = 1.0  # avreage SP variance is bounded with measurement precision for numeric stability
        
        self.var_min = 0.5 # variance lower bound = average variance x var_min
        self.var_max = 2.0 # variance upper bound = average variance x var_min
        
        self.cov_reg_weight = 0.2 # covariance is regularized with (1-lambda) * cov + lambda * diag(exp_area / 12)
        
        # neighbors defining connectedness from ls bit to ms bit (big endian)
        self.neighbor_x = [0, 1,  0, -1, 0,  1, -1, -1,  1]
        self.neighbor_y = [0, 0, -1,  0, 1, -1, -1,  1,  1]
        
        # juct connected look-up table
        self.LUT_JC = np.array([0,1,1,0,1,0,0,0,1,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,1,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,1,0,0,0,1,0,1,1,0], dtype=bool)
              
    def extract_superpixels(self, img_proc, img_disp = None, main_channel = 0):
        
        # get inputs, set process and display image
        self.img_proc = img_proc.copy().astype(np.float64)
        self.img_disp = img_disp
        self.main_channel = main_channel
        
        # get the size of the image
        self.height = self.img_proc.shape[0]
        self.width  = self.img_proc.shape[1]
        self.channels = 1 if self.img_proc.ndim == 2 else self.img_proc.shape[2]
        
        # set grid (coordinate) image
        self.img_grid = np.zeros((self.height, self.width, 2))
        self.img_grid[:,:,0], self.img_grid[:,:,1] = np.meshgrid(np.arange(0, self.width), np.arange(0, self.height))

        # initiate label image
        self.img_label = np.zeros((self.height, self.width), dtype=np.uint32)
        
        # set average SP area or number of required SPs
        if self.num_req_sps > 0:
            self.exp_area = self.height * self.width / float(self.num_req_sps)
        else:
            self.num_req_sps = np.round(self.height * self.width / self.exp_area)

        # set covariance regularization term
        self.cov_reg = np.eye(2) * self.exp_area / 12.0
        
        # compactness = 8 generates visually pleasing results for Lab color space & 16x16 SP size with no spatial or spectral normalization
        # if both cost fuctions are unnormalized, then both will be divided to default variance/covariance (multiplies given compactness by 0.93 for 16x16SP size)
        # if both cost funcions are bayesian, then no problem (these values are not used)
        # if one cost function is bayesian and the other is unnormalized then we need these default values!
        self.var_default = 4.0
        self.cov_default = self.exp_area / 12.0
        
        # perform initial tiling
        self.initial_tiling()

        # refine grid
        self.refine_grid()

    def initial_tiling(self):
        
        # perform initial tiling
        if self.tiling == 'iSQUARE':
            self.isquare_tiling()
        elif self.tiling == 'HEX':
            self.honeycomb_tiling()
        else:
            self.square_tiling()
    
        # if display image is specified, display initial tiling
        if self.img_disp is not None:
            plt.figure(dpi=300)
            plt.axis('off')
            plt.imshow(self.draw_boundaries(self.img_disp))
            plt.title('inital tiling')
            plt.show()
                
    def square_tiling(self):

        # edge length of the tiling square
        self.edge_length = np.sqrt(self.exp_area)

        # number of SPs on horizontal and vertical axes
        num_h = np.rint(self.width / self.edge_length).astype(int)
        num_v = np.rint(self.height / self.edge_length).astype(int)

        # initiate number of SPs and bbox
        self.num_sps = num_h * num_v
        self.bbox  = np.zeros((self.num_sps, 4), dtype=np.int32)
        
        # set column and row start indexes for SPs
        cst = np.rint(np.linspace(0, self.width, num_h + 1)).astype(int)
        rst = np.rint(np.linspace(0, self.height, num_v + 1)).astype(int)
        
        self.num_sps = 0

        # set label image and bounding box
        for j in range(num_v):
            for i in range(num_h):
                self.img_label[rst[j] : rst[j + 1], cst[i] : cst[i + 1]] = self.num_sps
                self.bbox[self.num_sps, :] = [cst[i], rst[j], cst[i + 1], rst[j + 1]] 
                self.num_sps += 1
        
    def honeycomb_tiling(self):
        
        # edge length of the tiling hexagon
        self.edge_length = np.sqrt(self.exp_area * 4 / (6 * np.sqrt(3)))
        
        # number of SPs on horizontal and vertical axes                     
        num_h = np.rint(self.width / (1.5 * self.edge_length)).astype(int)
        num_v = np.rint(self.height / (np.sqrt(3)/2 * self.edge_length)).astype(int)
        
        # spacing between initial SP centers
        horizontal_spacing = (self.width / num_h).astype(float)
        vertical_spacing = (self.height / num_v).astype(float)
        
        # centers
        x = horizontal_spacing / 2 + np.arange(num_h) * horizontal_spacing
        y = vertical_spacing / 2 + np.arange(num_v) * vertical_spacing
        
        # row&column start&end indexes of sp grid
        cst = np.floor(x - horizontal_spacing).astype(int); cst[0:2] = 0
        rst = np.floor(y - vertical_spacing).astype(int); rst[0:2] = 0
        cnd = np.ceil(x + horizontal_spacing).astype(int); cnd[-2:] = self.width
        rnd = np.ceil(y + vertical_spacing).astype(int); rnd[-2:] = self.height
        
        # initiate number of SPs, bbox and pixel to SP distance for each pixel
        self.num_sps = 0
        self.bbox = np.zeros((num_h * num_v, 4), dtype=np.int32)
        
        d_min = np.full(self.img_label.shape, np.inf)
        
        # set label image
        for j in range(num_v):
            for i in range(num_h): 
            
                # for even columns even rows, for odd columns odd rows will be set
                if not np.logical_xor(i % 2 == 0, j % 2 == 0):
                    continue
                    
                # image patch
                L = self.img_label[rst[j] : rnd[j], cst[i] : cnd[i]]
                X = self.img_grid[rst[j] : rnd[j], cst[i] : cnd[i], :]
                D = d_min[rst[j] : rnd[j], cst[i] : cnd[i]]
                
                # pixel to sp distance
                d = (X[:,:,0] - x[i]) ** 2 + (X[:,:,1] - y[j]) ** 2
                
                # replace label image and min distance if current sp distance is smaller than previously set
                mask = d < D
                
                L[mask] = self.num_sps
                D[mask] = d[mask]
                
                self.bbox[self.num_sps, :] = [cst[i], rst[j], cnd[i], rnd[j]] 
                
                self.num_sps += 1
                
        self.bbox = self.bbox[:self.num_sps, :]
                
    def isquare_tiling(self):
        
        # first perform square tiling to initiate bounding boxes
        self.square_tiling()
        
        # check the edge length, it must be an integer power of 2
        if self.edge_length != 16:
            print('Edge length must be 16 for iSQUARE tiling. Tiling is set tot SQAURE!')
            return
                        
        I0 = np.concatenate((np.expand_dims(self.img_proc[:, :, self.main_channel], 2), self.img_grid), axis=2)
        A0 = np.ones((I0.shape[0], I0.shape[1]), dtype=float)
        
        I1, A1, indR0, L1 = self.isq_downsample(I0, A0, spatial_reg=True)     
        I2, A2, indR1, L2 = self.isq_downsample(I1, A1, spatial_reg=True)       
        I3, A3, indR2, L3 = self.isq_downsample(I2, A2, spatial_reg=True)        
        I4, A4, indR3, L4 = self.isq_downsample(I3, A3, spatial_reg=True)
        
        L3 = L4.flatten()[indR3.astype(int)]
        L2 = L3.flatten()[indR2.astype(int)]
        L1 = L2.flatten()[indR1.astype(int)]
        L0 = L1.flatten()[indR0.astype(int)]
        
        self.img_label = L0.astype(int)
        
        self.bbox[:, 0] = np.maximum(self.bbox[:, 0] - 16, 0)
        self.bbox[:, 1] = np.maximum(self.bbox[:, 1] - 16, 0)
        self.bbox[:, 2] = np.minimum(self.bbox[:, 2] + 16, self.width)
        self.bbox[:, 3] = np.minimum(self.bbox[:, 3] + 16, self.height)
        
    def isq_downsample(self, inp_img, inp_area, spatial_reg = True):
    
        # get & check image size
        h  = inp_img.shape[0]
        w  = inp_img.shape[1]
        ch = inp_img.shape[2]
            
        if w % 2 or h % 2:
            print('Error: image must have even number of rows and columns!')
            return
        
        # check and apply spatial regularization
        if spatial_reg:
            G = inp_img[:,:,-2:].copy()
            
            G[:, 0::2, 0] = (G[:, 0::2, 0] + G[:, 1::2, 0]) / 2
            G[:, 1::2, 0] = G[:, 0::2, 0]
            G[0::2, :, 1] = (G[0::2, :, 1] + G[1::2, :, 1]) / 2
            G[1::2, :, 1] = G[0::2, :, 1]
        else:
            G = inp_img[:,:,-2:]
            
        # difference images
        diff_L = np.full([h+2, w+2], np.inf); # difference with left (right can be obtained with 1px horizontal shift)
        diff_T = np.full((h+2, w+2), np.inf); # difference with top (bottom can be obtained with 1px vertical shift)
        diff_LT = np.full((h+2, w+2), np.inf); # difference with left-top (right-bottom can be obtained with [1, 1]px shift)
        diff_RT = np.full((h+2, w+2), np.inf); # difference with right-top (left-bottom can be obtained with [-1, 1]px shift)
        
        # set difference images, all have same index with the input image
        diff_L[1:-1,2:-1] = np.sum((inp_img[:, 1:, :-2] - inp_img[:, :-1, :-2]) ** 2, axis=2) + np.sum((inp_img[:, 1:, -2:] - G[:, :-1, :]) ** 2, axis=2) 
        diff_T[2:-1,1:-1] = np.sum((inp_img[1:, :, :-2] - inp_img[:-1, :, :-2]) ** 2, axis=2) + np.sum((inp_img[1:, :, -2:] - G[:-1, :, :]) ** 2, axis=2)
        diff_LT[2:-1,2:-1] = np.sum((inp_img[1:, 1:, :-2] - inp_img[:-1, :-1, :-2]) ** 2, axis=2) + np.sum((inp_img[1:, 1:, -2:] - G[:-1, :-1, :]) ** 2, axis=2)
        diff_RT[2:-1,1:-2] = np.sum((inp_img[1:, :-1, :-2] - inp_img[:-1, 1:, :-2]) ** 2, axis=2) + np.sum((inp_img[1:, :-1, -2:] - G[:-1, 1:, :]) ** 2, axis=2)
        
        # error image to store 4 neighbor differences
        img_err = np.zeros([h // 2, w // 2, 4])
        
        # horizontal/vertical neighbor flags
        ind_h = np.zeros([h+2, w+2], dtype=bool)
        ind_v = np.zeros([h+2, w+2], dtype=bool)
        ind_d = np.zeros([h+2, w+2], dtype=np.uint8)
        
        # set 1px padded area
        A = np.ones([h+2, w+2]);  A[1:-1, 1:-1] = inp_area
    
        # alternative seed indexes on difference images (left-top, right-top, left-bottom, right-bottom) for downsampling
        L = [1, 2, 1, 2]
        T = [1, 1, 2, 2]
            
        # check start indexes on difference image for corresponding seed indexes
        X = [2, 1, 2, 1]
        Y = [2, 2, 1, 1]
        
        # initiate sum of squared errors for different seed indexes
        sse = np.zeros(4)
        
        # get sse for alternative seed indexes
        for i in range(4):
            
            # let values and area for pixels i and j are given as:
            # area[i] = N, value[i] = I
            # area[j] = M, value[j] = J
            # then when we merge i and j, mean = (N * I + M * J) / (M + N)
            # sse before merge:
            # sse[i] = sum(i^2) - N * I^2
            # sse[j] = sum(j^2) - M * J^2
            # sse after merge:
            # sse[i + j] = sum(i^2) + sum(j^2) - (N * I + M * J)^2 / (M + N)
            # sse increment = N * I^2 + M * J^2 - (N * I + M * J)^2 / (M + N)
            # sse increment = N * M / (N + M) * (I - J)^2
                        
            # horizontal downsampling
            W = A[T[i]:-1:2, X[i]:-1:2] # area of pixels to be merged to left or right
    
            WL = (W * A[T[i]:-1:2, X[i]-1:-2:2]) / (W + A[T[i]:-1:2, X[i]-1:-2:2]) # corresponding left/right weights
            WR = (W * A[T[i]:-1:2, X[i]+1:  :2]) / (W + A[T[i]:-1:2, X[i]+1:  :2])
            
            img_err[:,:,0] = WL * diff_L[T[i]:-1:2, X[i]:-1:2] # SSE for merging with left seed 
            img_err[:,:,1] = WR * diff_L[T[i]:-1:2, X[i]+1::2] # SSE for merging with right seed 
            
            ind_h[T[i]:-1:2, X[i]:-1:2] = img_err[:,:,1] < img_err[:,:,0]  # select left/right seed: 0 means left, 1 means right
            
            sse[i] = np.sum(img_err[:,:,0][~ind_h[T[i]:-1:2, X[i]:-1:2]]) + np.sum(img_err[:,:,1][ind_h[T[i]:-1:2, X[i]:-1:2]]) # initialize sse
    
            # vertical downsampling
            W = A[Y[i]:-1:2, L[i]:-1:2] # area of pixels to be merged to top or bottom
    
            WL = (W * A[Y[i]-1:-2:2, L[i]:-1:2]) / (W + A[Y[i]-1:-2:2, L[i]:-1:2]) # corresponding top/bottom weights
            WR = (W * A[Y[i]+1::2,   L[i]:-1:2]) / (W + A[Y[i]+1::2,   L[i]:-1:2])
            
            img_err[:,:,0] = WL * diff_T[Y[i]:-1:2, L[i]:-1:2] # SSE for merging with top seed 
            img_err[:,:,1] = WR * diff_T[Y[i]+1::2, L[i]:-1:2] # SSE for merging with bottom seed 
            
            ind_v[Y[i]:-1:2, L[i]:-1:2] = img_err[:,:,1] < img_err[:,:,0]  # select top/bottom seed: 0 means top, 1 means bottom
            
            sse[i] += np.sum(img_err[:,:,0][~ind_v[Y[i]:-1:2, L[i]:-1:2]]) + np.sum(img_err[:,:,1][ind_v[Y[i]:-1:2, L[i]:-1:2]]) # update sse
    
            # diagonal downsampling
            W = A[Y[i]:-1:2, X[i]:-1:2]
            WTL = (W * A[Y[i]-1:-2:2, X[i]-1:-2:2]) / (W + A[Y[i]-1:-2:2, X[i]-1:-2:2]) # top-left
            WTR = (W * A[Y[i]-1:-2:2, X[i]+1::2])   / (W + A[Y[i]-1:-2:2, X[i]+1::2])   # top-right
            WBR = (W * A[Y[i]+1::2,   X[i]+1::2])   / (W + A[Y[i]+1::2,   X[i]+1::2])   # bottom-right
            WBL = (W * A[Y[i]+1::2,   X[i]-1:-2:2]) / (W + A[Y[i]+1::2,   X[i]-1:-2:2]) # bottom-left
            
            
            # to merge with LT seed either top neighbor should merge to left or left neighbor merge with top, similar for other neighbors
            WTL[~np.logical_or(~ind_v[Y[i]:-1:2, X[i]-1:-2:2], ~ind_h[Y[i]-1:-2:2, X[i]:-1:2])] = np.inf 
            WTR[~np.logical_or(~ind_v[Y[i]:-1:2, X[i]+1:  :2],  ind_h[Y[i]-1:-2:2, X[i]:-1:2])] = np.inf
            WBL[~np.logical_or( ind_v[Y[i]:-1:2, X[i]-1:-2:2], ~ind_h[Y[i]+1:  :2, X[i]:-1:2])] = np.inf # cicik
            WBR[~np.logical_or( ind_v[Y[i]:-1:2, X[i]+1:  :2],  ind_h[Y[i]+1:  :2, X[i]:-1:2])] = np.inf
            
            img_err[:,:,0] = WTL * diff_LT[Y[i]:-1:2, X[i]:-1:2]    # SSE for merging with top-left seed 
            img_err[:,:,1] = WTR * diff_RT[Y[i]:-1:2, X[i]:-1:2]    # SSE for merging with top-right seed 
            img_err[:,:,2] = WBR * diff_LT[Y[i]+1::2, X[i]+1::2]    # SSE for merging with bottom-right seed 
            img_err[:,:,3] = WBL * diff_RT[Y[i]+1::2, X[i]-1:-2:2]  # SSE for merging with bottom-left seed 
            
            
            ind_d[Y[i]:-1:2, X[i]:-1:2] = np.argmin(img_err, axis=2)
            
            for n in range(4): sse[i] += np.sum(img_err[:,:,n][ind_d[Y[i]:-1:2, X[i]:-1:2] == n])
        
        # select the minimum error seed
        i = np.argmin(sse)
        
        # prepare input image for downsampling by weighting with input area
        inp_weighted = np.zeros([h+2, w+2, ch])
        inp_weighted[1:-1, 1:-1, :] = inp_img * np.expand_dims(inp_area, 2)
        
        # set area of image boundary to zero so they do not contribute to downsampled image
        A[[0, -1], :] = 0 
        A[:, [0, -1]] = 0
        
        # initiate output with seed
        out_img = inp_weighted[T[i]:-1:2, L[i]:-1:2, :].copy()
        out_area = A[T[i]:-1:2, L[i]:-1:2].copy()
        
        # initiate output inddex
        out_ind = np.zeros([h+2, w+2], dtype=np.uint32)
        out_ind[T[i]:-1:2, L[i]:-1:2] = np.arange(0, h//2 * w//2).reshape([h//2, w//2])
        out_label = np.arange(0, h//2 * w//2).reshape([h//2, w//2])
        
        # neighbors, indexes to be checked and required values to append
        neighbor_x = np.array([1,  0, -1, 0,  1, -1, -1,  1])
        neighbor_y = np.array([0, -1,  0, 1, -1, -1,  1,  1])
        
        ind_list = [ind_h, ind_v, ind_h, ind_v, ind_d, ind_d, ind_d, ind_d]
        req_val  = [0, 1, 1, 0, 3, 2, 1, 0]
        
        # add neighors
        for x, y, n in zip(neighbor_x + L[i], neighbor_y + T[i], np.arange(8)):
            
            mask = ind_list[n][y:y+h:2, x:x+w:2] == req_val[n]
            out_img[mask, :] += inp_weighted[y:y+h:2, x:x+w:2, :][mask]
            out_area[mask] += A[y:y+h:2, x:x+w:2][mask]
            
            out_ind[y:y+h:2, x:x+w:2][mask] = out_ind[T[i]:-1:2, L[i]:-1:2][mask]
            
        return out_img / np.expand_dims(out_area, 2), out_area, out_ind[1:-1, 1:-1], out_label
    
    def refine_grid(self):
        
        # set maximum number of iterations   
        '''
        if self.tiling == 'iSQUARE':
            self.max_iterations = np.maximum(np.ceil(self.edge_length * 0.4).astype(int), 4)
        elif self.tiling == 'HEX':
            self.max_iterations = np.ceil(self.edge_length).astype(int)
        else:
            self.max_iterations = np.ceil(self.edge_length * 0.8).astype(int)
        '''    
        self.max_iterations = np.ceil(self.edge_length).astype(int)
        
        # set image boundaries as sp = num_sps which does not exist! so they won't affect connectedness
        self.update_image_boundaries(value = self.num_sps)
        
        # initiate SP distributions
        self.update_sp_distributions()
        
        # set cost functions
        if self.spectral_cost == 'Bayesian':
            self.spectral_cost = self.spectral_bayesian
        else:
            self.spectral_cost = self.spectral_L2
        
        if self.spatial_cost == 'Bayesian':
            self.SpatialCost = self.spatial_bayesian
        else:
            self.SpatialCost = self.spatial_L2
            
        # refine label image
        for iteration in range(self.max_iterations):
                        
            print('iteration: ' + str(iteration))
            for i in np.arange(1, 4): # step by 3 pixels in each axis to preserve connectivity
                for j in np.arange(1, 4): # do not start from 0 as it has not 8 neighbors
                    self.refine_grid_iteration(i, j)
                    
            # update image boundaires with nearest neighbor at the end of the loop
            if iteration == self.max_iterations - 1:
                self.update_image_boundaries()
            
            # update sp distributions
            self.update_sp_distributions()
                    
            if self.img_disp is not None:
                plt.figure(dpi=300)
                plt.axis('off')
                plt.imshow(self.draw_boundaries(self.img_disp))
                plt.title('iter: ' + str(iteration))
                plt.show()
                            
    def refine_grid_iteration(self, l, t):
        
        # right and bottom boundaries, do not come to image boundaries as they have not 8 neighbors
        b = self.height - 1
        r = self.width - 1 
        
        # apply connectedness control and find the pixels can be updated
        B = np.zeros((np.ceil((b - t)/3).astype(int), np.ceil((r - l)/3).astype(int)), dtype=np.uint8)
        
        for n in np.arange(1, 9): #np.arange(8, 0, -1):
            B = np.left_shift(B, 1) + (self.img_label[t:b:3, l:r:3] == self.img_label[t+self.neighbor_y[n]:b+self.neighbor_y[n]:3, l+self.neighbor_x[n]:r+self.neighbor_x[n]:3]).astype(np.uint8)
          
        B = self.LUT_JC[B]
        
        # get the intensity values and coordinates of the pixels to be checked
        I = self.img_proc[t:b:3, l:r:3, :][B, :]        
        X = self.img_grid[t:b:3, l:r:3, :][B, :]
        
        # get the current labels and initiate pixel to sp distance
        labels_updated  = self.img_label[t:b:3, l:r:3][B]
                
        d_min = np.full(len(labels_updated), np.inf)
        
        
        # handle NaN's: if a pixel value or one of its candidate labels is NaN, then spectral distance for that pixel is not taken into account
        nan_mask = np.isnan(I)
        
        for n in np.arange(1,5): # check canddate labels of NaN
            
            # get neighbor label
            L = self.img_label[t+self.neighbor_y[n]:b+self.neighbor_y[n]:3, l+self.neighbor_x[n]:r+self.neighbor_x[n]:3][B]
            nan_mask = np.logical_or(nan_mask, np.isnan(self.mean[L, :]))
        
        I[nan_mask] = np.nan # set intensity values to NaN, as spectral distance is computed via nansum NaN channels won't be taken into account 
        
        # check neighbor sp distances
        for n in np.arange(1,5): 
            
            # get neighbor label
            L = self.img_label[t+self.neighbor_y[n]:b+self.neighbor_y[n]:3, l+self.neighbor_x[n]:r+self.neighbor_x[n]:3][B]
            
            # get distance to neighbor label
            d = self.spectral_cost(I, L) + self.compactness * self.SpatialCost(X, L)
            
            # piksels to be updated
            update = d < d_min
            
            #performa update
            labels_updated[update] = L[update]
            d_min[update] = d[update]
            
        # update label image
        self.img_label[t:b:3, l:r:3][B] = labels_updated
        
    def update_image_boundaries(self, value = None):
        
        if value == None:
            self.img_label[0, :] = self.img_label[1, :]
            self.img_label[:, 0] = self.img_label[:, 1]
            self.img_label[-1, :] = self.img_label[-2, :]
            self.img_label[:, -1] = self.img_label[:, -2]
            
            self.bbox[:, 0:1] -= 1
            self.bbox[:, 2:3] += 1
        else:
            self.img_label[0, :] = value
            self.img_label[:, 0] = value
            self.img_label[-1, :] = value
            self.img_label[:, -1] = value
        
    def update_sp_distributions(self):
        
        # spectral distribution is expressed as mean and variance, spatial distribution is expresed as center and covariance
        self.mean   = np.ones((self.num_sps+1, self.channels))
        self.var    = np.ones((self.num_sps+1, self.channels))
        self.center = np.ones((self.num_sps+1, 2))
        self.cov    = np.ones((self.num_sps+1, 2, 2))

        self.mean[self.num_sps, :] = np.inf
        self.center[self.num_sps, :] = np.inf
        
        # find sp distributions
        for n in range(self.num_sps):
            
            # extend current bbox
            l = np.maximum(self.bbox[n, 0] - 3, 0)
            t = np.maximum(self.bbox[n, 1] - 3, 0)
            r = np.minimum(self.bbox[n, 2] + 3, self.width)
            b = np.minimum(self.bbox[n, 3] + 3, self.height)
            
            # get mask for SP n
            M = self.img_label[t:b, l:r] == n
                     
            # pixels of SP n
            I = self.img_proc[t:b, l:r, :][M, :]
            X = self.img_grid[t:b, l:r, :][M, :]
           
            # set new bbox
            r = np.max(X[:, 0]) + 1
            l = np.min(X[:, 0])
            
            b = np.max(X[:, 1]) + 1
            t = np.min(X[:, 1])
            
            self.bbox[n, :] = np.array([l,t,r,b])
            
            # find spatial and spectral mean and covariance
            self.mean[n, :] = np.nanmean(I, 0)
            self.var[n, :]  = np.nanvar(I, 0)
    
            self.center[n, :] = np.nanmean(X, 0)
            
            self.cov[n, :, :] = np.cov(X.transpose())
            
        # bound variance INDEPENDENT CHANNELS
        '''
        var_avg = np.nanmean(self.var, 0)
        var_avg = np.maximum(var_avg, self.measurement_precision)
        
        var_limited = np.minimum(np.maximum(self.var, self.var_max * var_avg), self.var_min * var_avg)
        '''
        
        # bound variances, for each SP all channels are normalized with the same variance
        var_avg = np.nanmean(self.var)
        var_avg = np.maximum(var_avg, self.measurement_precision)
        
        var_limited = np.minimum(np.maximum(np.sum(self.var, 1, keepdims=True), self.var_max * var_avg), self.var_min * var_avg)
        
        # get variance inverse
        self.var_inv = 1 / var_limited
        self.var_log = np.log(var_limited)
        
        # regularize spatial covariance and get inverse
        covLimited = (1 - self.cov_reg_weight) * self.cov + self.cov_reg_weight * self.cov_reg
        
        covDet = covLimited[:, 0, 0] * covLimited[:, 1, 1] - covLimited[:, 1, 0] * covLimited[:, 0, 1]
                
        self.covInv = np.vstack((self.cov[:, 1, 1]/covDet, -self.cov[:, 1, 0]/covDet, self.cov[:, 0, 0]/covDet)).transpose()
        self.covLog = np.log(covDet)
        
    def spectral_L2(self, I, L):
        return np.nansum((I - self.mean[L, :]) ** 2, 1) / self.var_default

    def spectral_bayesian(self, I, L):
        return np.nansum((I - self.mean[L, :]) ** 2 * self.var_inv[L, :] + self.var_log[L, :], 1) 
    
    def spatial_L2(self, X, L):
        return np.sum((X - self.center[L, :]) ** 2, 1) / self.cov_default
    
    def spatial_bayesian(self, X, L):
        dx = X[:, 0] - self.center[L, 0]
        dy = X[:, 1] - self.center[L, 1]
        
        X = np.vstack((dx ** 2, dx * dy, dy ** 2)).transpose()
        
        return np.sum(X * self.covInv[L, :], axis=1) + self.covLog[L]
    
    def fill_mean_image(self):
        
        img_out = np.zeros(self.img_proc.shape)
        
        for n in np.arange(self.num_sps):
            
            b = self.bbox[n, :]
            
            mask = self.img_label[b[1]:b[3], b[0]:b[2]] == n
            
            img_out[b[1]:b[3], b[0]:b[2], :][mask, :] = self.mean[n, :]
            
        return img_out

    def draw_boundaries(self, I, color = [0, 0, 0]):

        # get label image
        L = self.img_label
        
        # initiate boundary image
        B = np.zeros((self.height, self.width), dtype=bool)

        # add right edge
        B[:, 0:-1] = np.logical_or(B[:, 0:-1], np.not_equal(L[:,0:-1], L[:,1:]));  
        # add right-bottom edge
        B[0:-1, 0:-1] = np.logical_or(B[0:-1, 0:-1], np.not_equal(L[0:-1, 0:-1], L[1:,1:])); 
        # add bottom edge  
        B[0:-1, :] = np.logical_or(B[0:-1, :], np.not_equal(L[0:-1, :], L[1:,:]));   

        # prepare output image
        J = I.copy()
        
        if J.ndim == 2:
            J = np.expand_dims(J, 2)

        for ch in range(J.shape[2]): J[B, ch] = color[ch] 
            
        return J









