"""PH 306 Week 2: Linear Algebra, Part 1.

Complete each function below. Use NumPy arrays for vector and matrix inputs and
outputs unless the function documentation specifies otherwise.
"""

# --- Imports --- #
# Built-in Libraries
from numpy.typing import NDArray
from typing import Any

# Numerical Libraries
import numpy as np
from astropy import units as u

# Local Utilities
from plotutil import colored_line_between_pts


# Type Hints
Array = NDArray[np.float64]


# Constants
EARTH_GRAVITY = 9.8 * u.m / u.s**2


def commutator(first: Array, second: Array) -> Array:
    """Compute the matrix commutator $[A, B] = AB - BA$."""
    return (first @ second) - (second @ first)

def are_perpendicular(first: Array, second: Array, tolerance=1e-10) -> bool:
    """Determine whether two vectors are perpendicular."""

    AB = first @ second

    return abs(AB) < tolerance 


def are_parallel(first: Array, second: Array, tolerance=1e-10) -> bool:
    """Determine whether two vectors are parallel."""

    dot_product = first @ second
    magnitude_product = np.linalg.norm(first) * np.linalg.norm(second)

    return abs(abs(dot_product) - magnitude_product) < tolerance


def are_commutative(first: Array, second: Array, tolerance=1e-10) -> bool:
    """Determine whether two matrices commute."""
    AB = first @ second
    BA = second @ first
    return abs(AB - BA).max() < tolerance

def is_hermitian(matrix: Array, tolerance=1e-10) -> bool:
    """Determine whether a matrix is Hermitian."""
    if matrix.shape[0] != matrix.shape[1]:
        return False  # Not a square matrix, cannot be Hermitian

    return abs(matrix - matrix.conj().T).max() < tolerance


def is_unitary(matrix: Array, tolerance=1e-10) -> bool:
    """Determine whether a matrix is unitary."""
    if matrix.shape[0] != matrix.shape[1]:
        return False  # Not a square matrix, cannot be unitary

    A_H = matrix.conj().T
    identity = np.eye(matrix.shape[0])
    return abs(matrix @ A_H - identity).max() < tolerance


def is_linear_operator(matrix: Array, tolerance=1e-10) -> bool:
    """Determine whether a matrix represents a linear operator."""
    u = np.random.rand(matrix.shape[1])
    v = np.random.rand(matrix.shape[1])
    return abs(matrix @ (u + v) - (matrix @ u + matrix @ v)).max() < tolerance


def projection(vector_a: Array, vector_b: Array) -> Array:
    """Project vector $\\vec{A}$ onto vector $\\vec{B}$."""
    dot_ab = vector_a @ vector_b
    dot_bb = vector_b @ vector_b
    return (dot_ab / dot_bb) * vector_b


def rotate_vector(vector: Array, axis: int, theta: float) -> Array:
    """Rotate a vector through angle theta about an axis in R^3."""

    if axis == 0:
        k = np.array([1.0, 0.0, 0.0])
    elif axis == 1:
        k = np.array([0.0, 1.0, 0.0])
    elif axis == 2:
        k = np.array([0.0, 0.0, 1.0])
    else:
        raise ValueError("Axis must be 0, 1, or 2.")

    v_cos = vector * np.cos(theta)
    v_sin = np.cross(k, vector) * np.sin(theta)
    v_dot = k * (k @ vector) * (1 - np.cos(theta))

    return v_cos + v_sin + v_dot



def plane_from_points(first: Array, second: Array, third: Array) -> tuple[Array, float]:
    """Find the plane through three noncollinear points."""
    if np.linalg.matrix_rank(np.vstack([first, second, third])) < 3:
        raise ValueError("The three points must be noncollinear.")
    u = second - first
    v = third - first
    normal = np.cross(u, v)
'''
Need to compute the offset (d) of the plane equation Ax + By + Cz + D = 0. The offset can be calculated using one of the points and the normal vector. The formula for the offset is:
d = - (A*x0 + B*y0 + C*z0)
'''


def distance_point_to_plane(point: Array, normal: Array, offset: float) -> float:
    """Find the minimum distance from a point to a plane."""
    raise NotImplementedError("Implement distance_point_to_plane")


def distance_between_lines(
        first_point: Array,
        first_direction: Array,
        second_point: Array,
        second_direction: Array) -> float:
    """Find the minimum distance between two lines in $\\mathbb{R}^3$."""
    raise NotImplementedError("Implement distance_between_lines")


def solve_cable_tension(N: int, L: float, rho: float, g: float = EARTH_GRAVITY) -> tuple[Array, Array]:
    """Solve for the tension in a hanging cable discretized into N segments."""
    raise NotImplementedError("Implement solve_cable_tension")


def plot_cable_tension(z: Array, T: Array, L: float) -> Any:
    """Plot the tension along a hanging cable, colored by tension magnitude."""
    raise NotImplementedError("Implement plot_cable_tension")
