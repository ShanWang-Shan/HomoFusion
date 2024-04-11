# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 21:46:28 2016

@author: chuong2
"""

import numpy as np
import cv2

def processZedInfo(info):
    ''' Parse camera parameters from ZED's parameter string
    '''
    Dict = {'Stereo': {}, 'LeftCam': {}, 'RightCam': {}}
    section = 'Stereo'
    for line in info.split('\n'):
        params = line.replace(' ', '').replace('\t', '').split(':')
        if params[0] == 'LeftCam':
            section = params[0]
            continue
        elif params[0] == 'RightCam':
            section = params[0]
            continue

        # convert to floating points or list of floating points
        if params[0] == 'disto':
            Dict[section][params[0]] = \
                [float(k) for k in params[1][1:-1].split(',')]
        else:
            Dict[section][params[0]] = float(params[1])

    Stereo = Dict['Stereo']
    T = np.array([Stereo['baseline'], Stereo['Ty'], Stereo['Tz']])
    Rx, _ = cv2.Rodrigues(np.array([Stereo['Rx'], 0, 0]))
    Ry, _ = cv2.Rodrigues(np.array([0, Stereo['convergence'], 0]))
    Rz, _ = cv2.Rodrigues(np.array([0, 0, Stereo['Rz']]))
    R = np.dot(Rx, np.dot(Ry, Rx))

    LeftCam = Dict['LeftCam']
    K_left = np.diag([LeftCam['fx'], LeftCam['fy'], 1])
    K_left[0, 2] = LeftCam['cx']
    K_left[1, 2] = LeftCam['cy']
    D_left = np.array(LeftCam['disto'])

    RightCam = Dict['LeftCam']
    K_right = np.diag([RightCam['fx'], RightCam['fy'], 1])
    K_right[0, 2] = RightCam['cx']
    K_right[1, 2] = RightCam['cy']
    D_right = np.array(RightCam['disto'])
    return {'T': T, 'R': R, 'K_left': K_left, 'D_left': D_left,
            'K_right': K_right, 'D_right': D_right}

def parseConfigFile(FileName):
    ''' Parse config file of ZED camera.
    Returns a dictionary'''
    ConfigDic = {}
    with open(FileName) as f:
        CamParams = {'cx': 0, 'cy': 0, 'fx': 0, 'fy': 0,
                     'k1': 0, 'k2': 0, 'p1': 0, 'p2': 0}
        SectionName = ''
        Side = ''
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            if line[0] == '[' and line[-1] == ']':
                SectionName = line[1:-1]
                if SectionName == 'STEREO':
                    Side = ''
                    ConfigDic[SectionName] = {}
                else:
                    Side = SectionName.split('_')[0]
                    SectionName = SectionName[len(Side)+1:]
                    if SectionName not in ConfigDic.keys():
                        ConfigDic[SectionName] = {}
                    ConfigDic[SectionName][Side] = CamParams.copy()
            else:
                parts = line.split('=')
                if SectionName == 'STEREO':
                    ConfigDic[SectionName][parts[0]] = float(parts[1])
                else:
                    ConfigDic[SectionName][Side][parts[0]] = float(parts[1])

    return ConfigDic


def getTransformFromConfig(FileName, Type='CAM_HD'):
    ''' Read config file and compute undistortion mapping function'''
    ConfigDic = parseConfigFile(FileName)
    Resolution = {'CAM_2K': [4416, 1242],
                  'CAM_FHD': [3840, 1080],
                  'CAM_HD': [2560, 720],
                  'CAM_VGA': [1344, 376]}
    Framerate = {'CAM_2K': [15],
                 'CAM_FHD': [30],
                 'CAM_HD': [30, 60],
                 'CAM_VGA': [100]}
    par1 = ConfigDic[Type]['LEFT']
    par2 = ConfigDic[Type]['RIGHT']
    par_stereo = ConfigDic['STEREO']
#    print(ConfigDic[Type])
#    print(ConfigDic['STEREO'])
    mat1 = np.array([[par1['fx'], 0, par1['cx']],
                     [0, par1['fy'], par1['cy']],
                     [0, 0, 1]])
    dis1 = np.array([par1['k1'], par1['k2'], 0, 0, 0])
    mat2 = np.array([[par2['fx'], 0, par2['cx']],
                     [0, par2['fy'], par2['cy']],
                     [0, 0, 1]])
    dis2 = np.array([par2['k1'], par2['k2'], 0, 0, 0])

    # rotation between left and right camera
    Rz, _ = cv2.Rodrigues(np.array([0, 0, par_stereo['RZ_HD']])) # from rotation Z vector to matrxi
    Ry, _ = cv2.Rodrigues(np.array([0, par_stereo['CV_HD'], 0]))
    Rx, _ = cv2.Rodrigues(np.array([par_stereo['RX_HD'], 0, 0]))
    R = np.dot(Rz, np.dot(Ry, Rx))

    T = np.array([-par_stereo['BaseLine'], 0, 0])
    
    imSize = tuple([int(Resolution[Type][0]/2), int(Resolution[Type][1])])

    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(mat1, dis1, mat2, dis2, imSize, R, T)
    map1x, map1y = cv2.initUndistortRectifyMap(mat1, dis1, R1, mat1, imSize, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(mat2, dis2, R2, mat1, imSize, cv2.CV_32FC1)

    return map1x, map1y, map2x, map2y, mat1, Q

def parseZedInfo(FileName):
    ''' Parse camera parameters from ZED's parameter string
    '''
    with open(FileName) as f:
        CamParams = {'cx': 0, 'cy': 0, 'fx': 0, 'fy': 0,
                     'k1': 0, 'k2': 0, 'p1': 0, 'p2': 0}
        Dict = {'Stereo': {}, 'LeftCam': {}, 'RightCam': {}}
        section = 'Stereo'
        for line in f:
            line = line.strip()
            params = line.replace(' ', '').replace('\t', '').split(':')
            if params[0] == 'LeftCam':
                section = params[0]
                Dict[section] = CamParams.copy()
                continue
            elif params[0] == 'RightCam':
                section = params[0]
                Dict[section] = CamParams.copy()
                continue

            # convert to floating points or list of floating points
            if params[0] == 'disto':
                print(params)
                Dict[section][params[0]] = \
                    [float(k) for k in params[1][1:-1].split(',')]
            else:
                Dict[section][params[0]] = float(params[1])
    return Dict

def getTransformFromInfo(FileName, Type='CAM_HD'):
    ''' Read config file and compute undistortion mapping function'''
    InfoDic = parseZedInfo(FileName)
    Resolution = {'CAM_2K': [4416, 1242],
                  'CAM_FHD': [3840, 1080],
                  'CAM_HD': [2560, 720],
                  'CAM_VGA': [1344, 376]}
    Framerate = {'CAM_2K': [15],
                 'CAM_FHD': [30],
                 'CAM_HD': [30, 60],
                 'CAM_VGA': [100]}
    par1 = InfoDic['LeftCam']
    par2 = InfoDic['RightCam']
    par_stereo = InfoDic['Stereo']
    print(par1)
    print(par2)
    print(par_stereo)
    mat1 = np.array([[par1['fx'], 0, par1['cx']],
                     [0, par1['fy'], par1['cy']],
                     [0, 0, 1]])
    dis1 = np.array(par1['disto'])
    mat2 = np.array([[par2['fx'], 0, par2['cx']],
                     [0, par2['fy'], par2['cy']],
                     [0, 0, 1]])
    dis2 = np.array(par2['disto'])

    Rz, _ = cv2.Rodrigues(np.array([0, 0, par_stereo['Rz']]))
    Ry, _ = cv2.Rodrigues(np.array([0, par_stereo['convergence'], 0]))
    Rx, _ = cv2.Rodrigues(np.array([par_stereo['Rx'], 0, 0]))
    R = np.dot(Rz, np.dot(Ry, Rx))
#    R = np.dot(Rx, np.dot(Ry, Rz))

    T = np.array([-par_stereo['baseline'], 0, 0])

    imSize = tuple([Resolution[Type][0]/2, Resolution[Type][1]])
    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(mat1, dis1, mat2, dis2, imSize, R, T)
    map1x, map1y = cv2.initUndistortRectifyMap(mat1, dis1, R1, mat1, imSize, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(mat2, dis2, R2, mat1, imSize, cv2.CV_32FC1)

    return map1x, map1y, map2x, map2y, Q

def getKRTInfo(FileName, Type='CAM_HD'):
    ''' Read config file and reutrn K matrix and T'''
    ConfigDic = parseConfigFile(FileName)
    Resolution = {'CAM_2K': [4416, 1242],
                  'CAM_FHD': [3840, 1080],
                  'CAM_HD': [2560, 720],
                  'CAM_VGA': [1344, 376]}
    Framerate = {'CAM_2K': [15],
                 'CAM_FHD': [30],
                 'CAM_HD': [30, 60],
                 'CAM_VGA': [100]}
    par1 = ConfigDic[Type]['LEFT']
    par2 = ConfigDic[Type]['RIGHT']
    par_stereo = ConfigDic['STEREO']
    #    print(ConfigDic[Type])
    #    print(ConfigDic['STEREO'])
    mat1 = np.array([[par1['fx'], 0, par1['cx']],
                     [0, par1['fy'], par1['cy']],
                     [0, 0, 1]])
    dis1 = np.array([par1['k1'], par1['k2'], 0, 0, 0])
    mat2 = np.array([[par2['fx'], 0, par2['cx']],
                     [0, par2['fy'], par2['cy']],
                     [0, 0, 1]])
    dis2 = np.array([par2['k1'], par2['k2'], 0, 0, 0])

    # rotation between left and right camera
    Rz, _ = cv2.Rodrigues(np.array([0, 0, par_stereo['RZ_HD']]))  # from rotation Z vector to matrxi
    Ry, _ = cv2.Rodrigues(np.array([0, par_stereo['CV_HD'], 0]))
    Rx, _ = cv2.Rodrigues(np.array([par_stereo['RX_HD'], 0, 0]))
    R = np.dot(Rz, np.dot(Ry, Rx))
    #    R = np.dot(Rx, np.dot(Ry, Rz))

    T = np.array([-par_stereo['BaseLine'], 0, 0])

    # imSize = tuple([int(Resolution[Type][0] / 2), int(Resolution[Type][1])])
    # R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(mat1, dis1, mat2, dis2, imSize, R, T)
    # map1x, map1y = cv2.initUndistortRectifyMap(mat1, dis1, np.eye(3), mat1, imSize, cv2.CV_32FC1)
    # map2x, map2y = cv2.initUndistortRectifyMap(mat2, dis2, np.eye(3), mat2, imSize, cv2.CV_32FC1)

    return mat1, mat2, R, T #, map1x, map1y, map2x, map2y


if __name__ == '__main__':
    ConfigDic = parseConfigFile('SN1994.conf')
    for key in ConfigDic.keys():
        print(key + ':')
        print(ConfigDic[key])