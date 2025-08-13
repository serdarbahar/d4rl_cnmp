import numpy as np
from scipy.spatial import ConvexHull


def sample_from_hull(hull, num_samples):

    """
    Generates uniform random samples from within a 2D convex hull.

    This is done by triangulating the hull, selecting triangles
    based on their area, and then finding a random point within
    the chosen triangle.

    Args:
        hull (scipy.spatial.ConvexHull): The convex hull to sample from.
        num_samples (int): The number of random points to generate.

    Returns:
        numpy.ndarray: An array of shape (num_samples, 2) of random points.
    """
    points = hull.points
    # The vertices of the hull in counter-clockwise order
    vertices = hull.vertices
    
    # Triangulate the convex hull by picking a central vertex (vertices[0])
    # and forming triangles with all other adjacent vertex pairs.
    triangles = []
    for i in range(1, len(vertices) - 1):
        triangles.append([vertices[0], vertices[i], vertices[i+1]])
    
    triangles = np.array(triangles)
    
    # Calculate the area of each triangle
    # Area = 0.5 * |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)|
    p1 = points[triangles[:, 0]]
    p2 = points[triangles[:, 1]]
    p3 = points[triangles[:, 2]]
    
    areas = 0.5 * np.abs(p1[:, 0] * (p2[:, 1] - p3[:, 1]) + 
                         p2[:, 0] * (p3[:, 1] - p1[:, 1]) + 
                         p3[:, 0] * (p1[:, 1] - p2[:, 1]))
                         
    # Calculate the probability of picking each triangle, proportional to its area
    total_area = np.sum(areas)
    probabilities = areas / total_area
    
    # Choose which triangle to sample from for each random point
    chosen_triangle_indices = np.random.choice(len(triangles), size=num_samples, p=probabilities)
    chosen_triangles = triangles[chosen_triangle_indices]
    
    # For each chosen triangle, generate a random point within it.
    # A random point in a triangle can be found with P = (1-sqrt(r1))A + sqrt(r1)(1-r2)B + sqrt(r1)r2*C
    # where r1, r2 are random numbers in [0,1] and A,B,C are the vertices.
    r1 = np.random.rand(num_samples, 1)
    r2 = np.random.rand(num_samples, 1)
    
    sqrt_r1 = np.sqrt(r1)
    
    # Points corresponding to the vertices of the chosen triangles
    A = points[chosen_triangles[:, 0]]
    B = points[chosen_triangles[:, 1]]
    C = points[chosen_triangles[:, 2]]
    
    random_samples = (1 - sqrt_r1) * A + sqrt_r1 * (1 - r2) * B + sqrt_r1 * r2 * C
    
    return random_samples