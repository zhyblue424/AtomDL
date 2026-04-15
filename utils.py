from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.feature import blob_log
from math import sqrt
import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torchvision.transforms import ToTensor
from sfin import SFIN
import numpy as np
from scipy import ndimage
from utils import *
import torchvision.transforms as T
from PIL import Image, ImageEnhance
import numpy as np
import argparse, os, time, random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.ndimage import maximum_filter, gaussian_filter
from scipy.spatial.distance import pdist, squareform
import random
from tqdm import tqdm
from itertools import combinations
from skimage.feature import blob_log


def sort_neighbors_clockwise(neighbors, center):
    rel = neighbors - center
    angles = np.arctan2(rel[:, 1], rel[:, 0])  # y,x
    sort_idx = np.argsort(angles)
    return neighbors[sort_idx]

def line_intersection(p1, p2, p3, p4):
    """Returns intersection point of lines (p1,p2) and (p3,p4)"""
    A = np.array([
        [p2[0] - p1[0], p3[0] - p4[0]],
        [p2[1] - p1[1], p3[1] - p4[1]]
    ])
    b = np.array([p3[0] - p1[0], p3[1] - p1[1]])
    if np.linalg.matrix_rank(A) < 2:
        return None  # Lines are parallel
    t, s = np.linalg.solve(A, b)
    intersection = p1 + t * (p2 - p1)
    return intersection

def atom_localization(X, Y, filtered_blobs, image_np, filtered_dist_matrix, coords_xy, highlight_groups, save_path_single, quantile = None, single_atom = True):
    """Determine the positions of atoms that satisfy the criteria, along with the positions of their six neighboring atoms"""
    for i, (y, x, r, intensity) in enumerate(tqdm(filtered_blobs, desc="Processing blobs")): 
        mask = (X - x) ** 2 + (Y - y) ** 2 <= (r * 0.8) ** 2 
        image_np[mask] = [255, 0, 0]
        top_k = 6
        indices = np.argsort(filtered_dist_matrix[i,:])
        neighbors = []
        for m in range(1, top_k + 1):
            x_n, y_n = coords_xy[indices[m]]
            neighbors.append((x_n, y_n))
            mask = (X - x_n) ** 2 + (Y - y_n) ** 2 <= (r * 0.8) ** 2 
            image_np[mask] = [255, 255, 0] 
        highlight_groups.append({
            'highlight': (x, y),
            'neighbors': neighbors
        })


    height, width = image_np.shape[:2]
    dpi = 100
    figsize = (width / dpi, height / dpi)
    plt.figure(figsize=figsize, dpi=dpi)
    plt.imshow(image_np)
    plt.axis('off')
    plt.tight_layout(pad=0)
    path = os.path.join(save_path_single, f"{100-quantile}highlight.png") if single_atom else os.path.join(save_path_single, "normal.png")
    plt.savefig(path, bbox_inches='tight', pad_inches=0) 

    

    """Determine the true atomic positions based on geometric relationships"""
    Intersection = []
    Dist = []
    Highlight = []
    Neighbors = []
    for idx, group in enumerate(highlight_groups):
        center = np.array(group['highlight'])
        neighbors = np.array(group['neighbors'])
        Highlight.append(center)
        sorted_neighbors = sort_neighbors_clockwise(neighbors, center)
        Neighbors.append(sorted_neighbors)
        symmetric_pairs = [(0, 3), (1, 4), (2, 5)]
        intersections = []
        for (i1, j1), (i2, j2) in combinations(symmetric_pairs, 2):
            pt1, pt2 = sorted_neighbors[i1], sorted_neighbors[j1]
            pt3, pt4 = sorted_neighbors[i2], sorted_neighbors[j2]
            intersection = line_intersection(pt1, pt2, pt3, pt4)
            if intersection is not None:
                intersections.append(intersection)
        intersection_point = np.mean(intersections, axis=0)  
        Intersection.append(intersection_point)
        plt.figure(figsize=(6, 6))
        plt.axis('equal')
        plt.gca().invert_yaxis()
        hexagon = np.vstack([sorted_neighbors, sorted_neighbors[0]])
        plt.plot(hexagon[:, 0], hexagon[:, 1], 'b--')
        plt.scatter(sorted_neighbors[:, 0], sorted_neighbors[:, 1], c='blue', label='Neighbor Points')

        for i, (x, y) in enumerate(sorted_neighbors):
            plt.text(x + 1, y, f'({x},{y})', color='blue', fontsize=8)

        plt.scatter(*center, c='red', marker='x', s=100, label='Highlight Point')
        plt.text(center[0]+1, center[1]-3, f'Highlight\n({center[0]:.1f},{center[1]:.1f})', color='red')

        for i, j in symmetric_pairs:
            pt1, pt2 = sorted_neighbors[i], sorted_neighbors[j]
            plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'purple', linewidth=1)

        if intersection_point is not None:
            plt.scatter(*intersection_point, c='green', marker='o', s=80, label='Line Intersect')

        plt.scatter(*intersection_point, c='green', marker='o', s=80, label='Intersect')
        plt.text(intersection_point[0] + 1, intersection_point[1] + 1,
                f'Intersect\n({intersection_point[0]:.1f},{intersection_point[1]:.1f})',
                color='green', fontsize=9)

        plt.plot([center[0], intersection_point[0]], [center[1], intersection_point[1]], 'k:', label='Offset to Center')

        plt.title("Symmetric Line Intersections vs Highlight Point")
        plt.gca().invert_yaxis()
        plt.grid(True)
        plt.legend()
        path = os.path.join(save_path_single, f'{100-quantile}highlight', f'intersection{idx}.png') if single_atom else os.path.join(save_path_single, 'normal', f'normal{idx}.png')
        plt.savefig(path, bbox_inches='tight', pad_inches=0)

    import csv
    path = os.path.join(save_path_single, f'{100-quantile}highlight.csv') if single_atom else path = os.path.join(save_path_single, 'normal.csv')
    with open(path, mode='w', newline='') as file:
        writer = csv.writer(file)
        header = ['center_x', 'center_y']
        header += ['inter_x', 'inter_y']
        for i in range(6):
            header += [f'n{i}_x', f'n{i}_y']
        writer.writerow(header)
        for center, group, inter in zip(Highlight, Neighbors, Intersection):
            row = list(center)
            row += list(inter)
            for neighbor in group:
                row += list(neighbor)
            writer.writerow(row)
    plt.close('all')