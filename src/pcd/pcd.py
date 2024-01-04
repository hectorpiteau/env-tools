"""
Author: Hector Piteau (hector.piteau@gmail.com)
pcd.py (c) 2023
Desc: Point Cloud GMM
Created:  2023-12-26T15:43:17.105Z
Modified: 2023-12-26T15:50:04.186Z
"""

import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
from skspatial.objects import Sphere
import time 
from typing import List
import sys

# from pycave.clustering import GaussianMixtureModel

def normalize_gaussian_indep(mean : np.array, std : np.array):
    """Express the gaussian with a mean expressed with respect to the 
    barycenter of all means. 
    And rearrange the standard deviations where first dimensions are
    the largest. 
    
    Args:
        mean (np.array): _description_
        std (np.array): _description_
    """
    center = np.mean(mean, axis=0)
    
    

def kl_mvn(m0, S0, m1, S1):
    """
    Kullback-Liebler divergence from Gaussian pm,pv to Gaussian qm,qv.
    Also computes KL divergence from a single Gaussian pm,pv to a set
    of Gaussians qm,qv.
    

    From wikipedia
    KL( (m0, S0) || (m1, S1))
         = .5 * ( tr(S1^{-1} S0) + log |S1|/|S0| + 
                  (m1 - m0)^T S1^{-1} (m1 - m0) - N )
    """
    # store inv diag covariance of S1 and diff between means
    N = m0.shape[0]
    iS1 = np.linalg.inv(S1)
    diff = m1 - m0

    # kl is made of three terms
    tr_term   = np.trace(iS1 @ S0)
    det_term  = np.log(np.linalg.det(S1)/np.linalg.det(S0)) #np.sum(np.log(S1)) - np.sum(np.log(S0))
    quad_term = diff.T @ np.linalg.inv(S1) @ diff #np.sum( (diff*diff) * iS1, axis=1)
    #print(tr_term,det_term,quad_term)
    return .5 * (tr_term + det_term + quad_term - N) 

def compare_gaussian_std(stds_a : List, stds_b : List, method = "ED") -> int:
    """_summary_

    Args:
        stds_a (List): _description_
        stds_b (List): _description_
        method (str, optional): _description_. Defaults to "ED".

    Returns:
        int: 0-1 : 0 not the same, 1 exactly the same.
    """
    a = stds_a.copy().sort()
    b = stds_b.copy().sort()
    res = 0
    
    if method == "COS":
        """
            Normalized Cosine similarity score. Widely used but has several drawbacks.
            
            - sensitivity to vector length.
            - captures only linear relationships.
            - sensitive to outliers.            
        """ 
        dt = np.dot(a,b.T)
        
        if np.linalg.norm(a) == 0.0 or np.linalg.norm(b) == 0.0:
            return 0.0
        
        res = (dt) / (np.linalg.norm(a) * np.linalg.norm(b))
        res = (res + 1) / 2.0
    
    return res
    

def compare_gaussian_list(gla : List, glb : List):
    """Brute force comparison.

    Args:
        gla (List): _description_
        glb (List): _description_
    """
    similarity_matrix = np.zero((len(gla), len(glb)))
    
    # can be parallelized.
    for i in range(0, len(gla)):
        for j in range(0, len(glb)):
            similarity_matrix[i,j] = compare_gaussian_std(gla[i], glb[j])
    
    return similarity_matrix

def random_rotation_matrix():
    # Generate a random axis
    axis = np.random.rand(3)
    axis /= np.linalg.norm(axis)

    # Generate a random angle (in radians)
    angle = np.random.uniform(0, 2 * np.pi)

    # Create the rotation matrix using the axis-angle representation
    c = np.cos(angle)
    s = np.sin(angle)
    t = 1 - c
    x, y, z = axis
    rotation_matrix = np.array([
        [t*x*x + c, t*x*y - s*z, t*x*z + s*y],
        [t*x*y + s*z, t*y*y + c, t*y*z - s*x],
        [t*x*z - s*y, t*y*z + s*x, t*z*z + c]
    ])

    return rotation_matrix

def random_translation_vector():
    # Generate a random translation vector
    translation = np.random.rand(3)

    return translation

def homogeneous_transformation_matrix(rotation_matrix, translation_vector):
    # Create a 4x4 transformation matrix
    transformation_matrix = np.eye(4)
    
    # Assign rotation matrix to the top-left 3x3 submatrix
    transformation_matrix[:3, :3] = rotation_matrix
    
    # Assign translation vector to the rightmost column
    transformation_matrix[:3, 3] = translation_vector
    
    return transformation_matrix

def load_model():
    rotation_matrix = random_rotation_matrix()
    translation_vector = random_translation_vector()

    transformation_matrix = homogeneous_transformation_matrix(rotation_matrix, translation_vector)

    file_path = "/home/hepiteau/Work/DRTMCVFX/SuperTensoRF/test-tiny-tensorf/data/ModelNet10/monitor/test/monitor_0466.off"
    mesh = o3d.io.read_triangle_mesh(file_path)

    num_points = 30000 # You can adjust the number of points
    pcd1 = mesh.sample_points_uniformly(number_of_points=num_points)
    pcd2 = mesh.sample_points_uniformly(number_of_points=num_points)
    pcd2.transform(transformation_matrix)

    return pcd1, pcd2, num_points, transformation_matrix

def compute_gmm(pcd, components=5, filename="test.png",  show=True):
    # Generate a synthetic 3D point cloud
    np.random.seed(42)

    # Reshape the point cloud to match the input requirements of GaussianMixture
    point_cloud = pcd

    start_time = time.time()
    # Fit a Gaussian Mixture Model
    n_components = components # Number of components (gaussians)
    gmm = GaussianMixture(n_components=n_components, covariance_type='full')
    
    # gmm_pycave = pycave.bayes.GaussianMixtureModel()
    # gmm_pycave.fit(point_cloud)
    gmm.fit(point_cloud)

    # Predict the component labels for each point in the point cloud
    labels = gmm.predict(point_cloud)

    # Get the means and covariances of the fitted gaussians
    means = gmm.means_
    covariances = gmm.covariances_

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"GMM time taken: {elapsed} sec. N_components: {n_components}")

    if not show:
        # print(means)
        # print(covariances)
        return
    
    # Plot the original point cloud and the fitted gaussians
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], c=labels, cmap='viridis', marker='o')

    # Plot the fitted gaussians
    for i in range(n_components):
        mean = means[i]
        cov = covariances[i]
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        scaling_factor = 2.0  # You can adjust the scaling factor based on your preference
        scaled_eigenvectors = scaling_factor * np.sqrt(eigenvalues[:, None]) * eigenvectors
        ax.quiver(mean[0], mean[1], mean[2], scaled_eigenvectors[0, 0], scaled_eigenvectors[1, 0], scaled_eigenvectors[2, 0], color='red')
        ax.quiver(mean[0], mean[1], mean[2], scaled_eigenvectors[0, 1], scaled_eigenvectors[1, 1], scaled_eigenvectors[2, 1], color='green')
        ax.quiver(mean[0], mean[1], mean[2], scaled_eigenvectors[0, 2], scaled_eigenvectors[1, 2], scaled_eigenvectors[2, 2], color='blue')
        # r = 10
        # u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]
        # x = np.cos(u) * np.sin(v)
        # y = np.sin(u) * np.sin(v)
        # z = np.cos(v)
        # x = x * np.linalg.norm(scaled_eigenvectors[:, 0]) + mean[0]
        # y = y * np.linalg.norm(scaled_eigenvectors[:, 1]) + mean[1]
        # z = z * np.linalg.norm(scaled_eigenvectors[:, 2]) + mean[2]
        # ax.plot_surface(x,y,z, alpha=0.2)
    
    
    # sphere = Sphere([1, 2, 3], 2)
    # sphere.plot_3d(ax, alpha=0.2)
    # sphere.point.plot_3d(ax, s=100)
    
    plt.show()
    # plt.savefig(filename)


def test():
    pcd1, pcd2, num_points, transformation_matrix = load_model()
    # Load the point cloud from an .off file

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    pcd1_pts = np.asarray(pcd1.points)
    pcd2_pts = np.asarray(pcd2.points)
    # ax.scatter(pcd2_pts[:, 0], pcd2_pts[:, 1], pcd2_pts[:, 2], cmap='viridis', marker='o')
    # ax.scatter(pcd1_pts[:, 0], pcd1_pts[:, 1], pcd1_pts[:, 2], cmap='viridis', marker='o')
    # plt.show()

    tab_a = []
    for i in range(5,100):
        tab_a.append(compute_gmm(pcd1_pts, components=i, filename="Figure_1.png", show=False))
    print(tab_a)
    # compute_gmm(pcd2_pts, "Figure_2.png", show=False)



def run_exp() -> int:
    # Extract gaussian features N times for the same model.
    N = 1000
    nb_components = 10
    
    pcd1, pcd2, num_points, transformation_matrix = load_model()
    
    tab = []
    for i in range(0, N):
        res = compute_gmm(pcd1, components=nb_components, show=False)
        tab.append(res)
        
    return 0

def main() -> int:
    return run_exp()

if __name__ == "__main__":
    sys.exit(main())