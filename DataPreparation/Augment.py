import numpy as np
# import volumentations as V
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import open3d as o3d
import random
import glob
import math

def generate_rotation_x_matrix(theta):
    mat = np.eye(3, dtype=np.float32)
    mat[1, 1] = math.cos(theta)
    mat[1, 2] = -math.sin(theta)
    mat[2, 1] = math.sin(theta)
    mat[2, 2] = math.cos(theta)
    return mat


def generate_rotation_y_matrix(theta):
    mat = np.eye(3, dtype=np.float32)
    mat[0, 0] = math.cos(theta)
    mat[0, 2] = math.sin(theta)
    mat[2, 0] = -math.sin(theta)
    mat[2, 2] = math.cos(theta)
    return mat


def generate_rotation_z_matrix(theta):
    mat = np.eye(3, dtype=np.float32)
    mat[0, 0] = math.cos(theta)
    mat[0, 1] = -math.sin(theta)
    mat[1, 0] = math.sin(theta)
    mat[1, 1] = math.cos(theta)
    return mat


def generate_random_rotation_matrix(angle1=-90, angle2=90):
    thetax = np.random.uniform(np.pi * angle1 / 180.0,np.pi * angle2 / 180.0)
    thetay = np.random.uniform(np.pi * angle1 / 180.0,np.pi * angle2 / 180.0)
    thetaz = np.random.uniform(np.pi * angle1 / 180.0,np.pi * angle2 / 180.0)
    matx = generate_rotation_x_matrix(thetax)
    maty = generate_rotation_y_matrix(thetay)
    matz = generate_rotation_z_matrix(thetaz)
    return np.dot(matx, np.dot(maty, matz))

def generate_random_tranlation_vector(range1=-1.5, range2=1.5):
    tranlation_vector = np.random.uniform(range1, range2, size=(3, )).astype(np.float32)
    return tranlation_vector


def transform(pc, R=None, t=None):
    if R is not None:
        pc = np.dot(pc, R.T)
    if t is not None:
        pc = pc + t
    return pc

def jitter_point_cloud(pc, sigma=0.001, clip=0.005):
    N, C = pc.shape
    assert(clip > 0)
    #jittered_data = np.clip(sigma * np.random.randn(N, C), -1*clip, clip).astype(np.float32)
    jittered_data = np.clip(
        np.random.normal(0.0, scale=sigma, size=(N, 3)),
        -1 * clip, clip).astype(np.float32)
    jittered_data += pc
    return jittered_data

def scale_to_unit_cube(point_cloud):
    """
    Scales a point cloud to fit within a unit cube.

    Args:
        point_cloud: a numpy array of shape (N, 3) representing a point cloud.

    Returns:
        A numpy array of the same shape, scaled to fit within a unit cube.
    """
    # find the minimum and maximum values of the point cloud
    p_min = point_cloud.min(axis=0)
    p_max = point_cloud.max(axis=0)

    # calculate the scaling factor
    scale = np.max(p_max - p_min)

    # subtract the minimum value from all points
    point_cloud -= p_min

    # divide all points by the scaling factor
    point_cloud /= scale

    return point_cloud