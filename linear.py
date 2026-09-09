"""PH 306 Week 2: Linear Algebra, Part 1.

Complete each function below. Use NumPy arrays for vector and matrix inputs and
outputs unless the function documentation specifies otherwise.
"""

# --- Imports --- #
# Built-in Libraries
from matplotlib.collections import LineCollection
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from typing import Any, Callable

# Numerical Libraries
import numpy as np
from astropy import units as u

# Local Utilities
# (from plotutil import colored_line_between_pts) is not used


# Type Hints
Array = NDArray[np.float64]


# Constants
EARTH_GRAVITY = 9.8 * u.m / u.s**2


def commutator(first: Array, second: Array) -> Array:
    """Compute the matrix commutator $[A, B] = AB - BA$.

    Parameters
    ----------
    first : Array
        The first matrix.
    second : Array
        The second matrix.

    Returns
    -------
    Array
        The commutator [A, B].
    """
    return (first @ second) - (second @ first)


def are_perpendicular(first: Array, second: Array, tolerance: float = 1e-10) -> bool:
    """Determine whether two vectors are perpendicular.

    Parameters
    ----------
    first : Array
        The first vector.
    second : Array
        The second vector.
    tolerance : float, optional
        The tolerance for determining perpendicularity. Default is 1e-10.

    Returns
    -------
    bool
        True if the vectors are perpendicular, False otherwise.
    """

    AB = first @ second

    return bool(np.abs(AB) < tolerance)


def are_parallel(first: Array, second: Array, tolerance: float = 1e-10) -> bool:
    """Determine whether two vectors are parallel.

    Parameters
    ----------
    first : Array
        The first vector.
    second : Array
        The second vector.
    tolerance : float, optional
        The tolerance for determining parallelism. Default is 1e-10.

    Returns
    -------
    bool
        True if the vectors are parallel, False otherwise.
    """
    if (
        np.isclose(np.linalg.norm(first), 0.0)
        or np.isclose(np.linalg.norm(second), 0.0)
    ):
        raise ValueError("Cannot determine parallelism for zero vectors.")

    dot_product = first @ second
    magnitude_product = np.linalg.norm(first) * np.linalg.norm(second)

    return bool(abs(abs(dot_product) - magnitude_product) < tolerance)


def are_commutative(first: Array, second: Array, tolerance: float = 1e-10) -> bool:
    """Determine whether two matrices commute.

    Parameters
    ----------
    first : Array
        The first matrix.
    second : Array
        The second matrix.
    tolerance : float, optional
        The tolerance for determining commutativity. Default is 1e-10.

    Returns
    -------
    bool
        True if the matrices are commutative, False otherwise.
    """
    AB = first @ second
    BA = second @ first
    return bool(abs(AB - BA).max() < tolerance)


def is_hermitian(matrix: Array, tolerance: float = 1e-10) -> bool:
    """Determine whether a matrix is Hermitian.

    Parameters
    ----------
    matrix : Array
        The matrix to check.
    tolerance : float, optional
        The tolerance for determining Hermiticity. Default is 1e-10.

    Returns
    -------
    bool
        True if the matrix is Hermitian, False otherwise.
    """
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Matrix must be square to check for Hermiticity.")

    return bool(abs(matrix - matrix.conj().T).max() < tolerance)


def is_unitary(matrix: Array, tolerance: float = 1e-10) -> bool:
    """Determine whether a matrix is unitary.

    Parameters
    ----------
    matrix : Array
        The matrix to check.
    tolerance : float, optional
        The tolerance for determining unitarity. Default is 1e-10.

    Returns
    -------
    bool
        True if the matrix is unitary, False otherwise.
    """
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Matrix must be square to check for unitarity.")

    A_H = matrix.conj().T
    identity = np.eye(matrix.shape[0])
    return bool(abs(matrix @ A_H - identity).max() < tolerance)


def is_linear_operator(matrix: Array, tolerance: float = 1e-10) -> bool:
    """Determine whether a matrix represents a linear operator.

    Parameters
    ----------
    matrix : Array
        The matrix to check.
    tolerance : float, optional
        The tolerance for determining linearity. Default is 1e-10.

    Returns
    -------
    bool
        True if the matrix represents a linear operator, False otherwise.
    """
    if matrix.ndim != 2:
        return False

    return matrix.shape[0] == matrix.shape[1]


def projection(vector_a: Array, vector_b: Array) -> Array:
    """Project vector $\\vec{A}$ onto vector $\\vec{B}$.

    Parameters
    ----------
    vector_a : Array
        The vector to be projected.
    vector_b : Array
        The vector onto which to project.

    Returns
    -------
    Array
        The projection of vector_a onto vector_b.
    """
    dot_ab = vector_a @ vector_b
    dot_bb = vector_b @ vector_b
    return (dot_ab / dot_bb) * vector_b


def rotate_vector(vector: Array, axis: int, theta: float) -> Array:
    """Rotate a vector through angle theta about an axis in R^3.

    Parameters
    ----------
    vector : Array
        The vector to rotate.
    axis : int
        The axis about which to rotate (0, 1, or 2).
    theta : float
        The angle of rotation in radians.

    Returns
    -------
    Array
        The rotated vector.
    """

    if vector.shape != (3,):
        raise ValueError("Vector must be a 3D vector.")

    if axis not in (0, 1, 2):
        raise ValueError("Axis must be 0, 1, or 2.")

    rotated = vector.copy()

    indices = [i for i in range(3) if i != axis]
    i, j = indices

    c = np.cos(theta)
    s = np.sin(theta)

    rotated[i] = vector[i] * c - vector[j] * s
    rotated[j] = vector[i] * s + vector[j] * c

    return rotated


def plane_from_points(first: Array, second: Array, third: Array) -> tuple[Array, float]:
    """Find the plane through three noncollinear points.

    Parameters
    ----------
    first : Array
        The first point.
    second : Array
        The second point.
    third : Array
        The third point.

    Returns
    -------
    tuple[Array, float]
        The normal vector and offset of the plane.
    """

    u = second - first
    v = third - first
    normal_cross = np.cross(u, v)

    if np.isclose(np.linalg.norm(normal_cross), 0):
        raise ValueError("The three points are collinear and do not define a plane.")

    normal = normal_cross / np.linalg.norm(normal_cross)
    offset = float(normal @ first)
    return normal, offset


def distance_point_to_plane(point: Array, normal: Array, offset: float) -> float:
    """Find the minimum distance from a point to a plane.

    Parameters
    ----------
    point : Array
        The point from which to find the distance.
    normal : Array
        The normal vector of the plane.
    offset : float
        The offset of the plane.

    Returns
    -------
    float
        The minimum distance from the point to the plane.
    """
    distance = abs(normal @ point - offset) / np.linalg.norm(normal)
    return float(distance)


def distance_between_lines(
        point1: Array,
        direction1: Array,
        point2: Array,
        direction2: Array
        ) -> float:
    """Find the minimum distance between two lines in R^3.

    Parameters
    ----------
    point1 : Array
        A point on the first line.
    direction1 : Array
        The direction vector of the first line.
    point2 : Array
        A point on the second line.
    direction2 : Array
        The direction vector of the second line.

    Returns
    -------
    float
        The minimum distance between the two lines.
    """

    cross = np.cross(direction1, direction2)
    difference = point2 - point1

    if np.isclose(np.linalg.norm(cross), 0):
        return float(
            np.linalg.norm(np.cross(difference, direction1))
            / np.linalg.norm(direction1)
        )

    return float(
        abs(difference @ cross) / np.linalg.norm(cross)
    )


def solve_cable_tension(
        N: int,
        L: float,
        rho: Callable[[Any], Any],
        g: float = EARTH_GRAVITY
        ) -> tuple[Any, Any]:
    """Solve for the tension in a hanging cable discretized into N segments.

    Parameters
    ----------
    N : int
        The number of segments.
    L : float
        The length of the cable.
    rho : Callable[[Any], Any]
        The density of the cable as a function of position.
    g : float, optional
        The acceleration due to gravity (default is EARTH_GRAVITY).

    Returns
    -------
    tuple[Any, Any]
        The positions and tensions along the cable.
    """
    delta_z = L / N

    z = np.linspace(0.0, 1.0, N + 1) * L

    z_mid = (z[:-1] + z[1:]) / 2

    A = (np.diag(np.ones(N)) + np.diag(-np.ones(N - 1), k=-1))

    density = np.ones(N) * rho(z_mid)

    b = density * g * delta_z
    if hasattr(b, "si"):
        tension = np.linalg.solve(A, b.si.value)
        T = tension * u.N
    else:
        tension = np.linalg.solve(A, b)
        T = tension * u.N

    return z, T


def plot_cable_tension(z: Array, T: Array, L: float) -> Any:
    """Plot the tension along a hanging cable, colored by tension magnitude.

    Parameters
    ----------
    z : Array
        The positions along the cable.
    T : Array
        The tensions along the cable.
    L : float
        The length of the cable.

    Returns
    -------
    Any
        The matplotlib figure and axes.
    """
    if hasattr(z, "value"):
        z_values = z.value
    else:
        z_values = np.asarray(z)

    if hasattr(T, "value"):
        T_values = T.value
    else:
        T_values = np.asarray(T)

    if hasattr(L, "value"):
        L_value = L.value
    else:
        L_value = L

    x = np.zeros_like(z_values)

    points = np.column_stack((x, z_values))
    segments = np.stack((points[:-1], points[1:]), axis=1)

    line = LineCollection(
        segments.tolist(),
        cmap="inferno_r",
        array=T_values
    )

    fig, ax = plt.subplots()

    ax.add_collection(line)

    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, L_value + 0.5)

    ax.set_xticks([])

    cbar = fig.colorbar(line, ax=ax)
    cbar.set_label("Tension (N)")

    ax.set_ylabel("Height (m)")
    ax.set_title(
        f"Tension in a Hanging Cable with {len(T_values)} Segments"
    )

    return fig, ax
