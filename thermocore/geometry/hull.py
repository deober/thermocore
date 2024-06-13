import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull
from typing import List, Tuple, Sequence
from math import factorial

def barycentric_coordinates(point: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """Returns the barycentric coordinates of `point` with respect to the simplex defined by `vertices`.

    TODO: Take multiple points?

    Parameters
    ----------
    point : np.ndarray of floats, shape (n_dim,)
        Point to get barycentric coordinates for.
    vertices : np.ndarray of floats, shape (n_dim + 1, n_dim)
        Vertices of reference simplex for the barycentric coordinates.

    Returns
    -------
    np.ndarray of floats, shape (n_dim + 1,)
        Barycentric coordinates of `point`.
    """
    # Check dimensions
    n_dim = len(point)
    if not vertices.shape[0] == n_dim + 1:
        raise ValueError("Number of vertices provided inconsistent with dimension.")

    if not vertices.shape[1] == n_dim:
        raise ValueError("Point and vertex dimensions inconsistent.")

    # Find barycentric coordinates
    H = np.vstack((vertices.transpose(), np.ones((1, n_dim + 1))))
    H_inv = np.linalg.inv(H)
    return H_inv @ np.append(point, 1)


def inside_convex_hull(points: np.ndarray, test_points: np.ndarray) -> List[bool]:
    """Returns a list of booleans indicating whether each point in `test_points` is inside the convex hull of `points`.

    This does not require finding the convex hull of `points`, only determining whether each of `test_points`
    can be expressed as a convex combination of `points`, which can be done using linear programming.

    For a single test point p and the points x_i, we check whether there exist coefficients a_i such that
    p = a_1*x_1 + a_2*x_2 + ... + a_n*x_n, where the a_i are non-negative and sum to 1.

    Adapted from: https://stackoverflow.com/a/43564754

    Parameters
    ---------
    points : np.ndarray of floats, shape (n_points, n_dim)
        Points defining the convex hull.
    test_points : np.ndarray of floats, shape (n_test_points, n_dim)
        Points to be tested for whether they are inside the convex hull.

    Returns
    -------
    list[bool]
        Booleans indicating whether each test point is inside the convex hull.
    """
    n_points = len(points)
    c = np.zeros(n_points)
    A = np.vstack((points.transpose(), np.ones((1, n_points))))
    return [linprog(c, A_eq=A, b_eq=np.append(p, 1.0)).success for p in test_points]


def full_hull(
    compositions: np.ndarray, energies: np.ndarray, qhull_options=None
) -> ConvexHull:
    """Returns the full convex hull of the points specified by appending `energies` to `compositions`.

    Parameters
    ----------
    compositions: np.ndarray of floats, shape (n_points, n_composition_axes)
        Compositions of points.
    energies: np.ndarray of floats, shape (n_points,)
        Energies of points.
    qhull_options: str
        Additional optionals that can be passed to Qhull. See details on the scipy.spatial.ConvexHull documentation. Default=None
    Returns
    -------
    ConvexHull
        Convex hull of points.
    """
    return ConvexHull(
        np.hstack((compositions, energies[:, np.newaxis])), qhull_options=qhull_options
    )


def lower_hull(
    convex_hull: ConvexHull, tolerance: float = 1e-14
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns the vertices and simplices of the lower convex hull (with respect to the last coordinate) of `convex_hull`.

    Parameters
    ----------
    convex_hull : ConvexHull
        Complete convex hull object.
    tolerance : float, optional
        Tolerance for identifying lower hull simplices (default is 1e-14).

    Returns
    -------
    lower_hull_vertex_indices : np.ndarray of ints, shape (n_vertices,)
        Indices of points forming the vertices of the lower convex hull.
    lower_hull_simplex_indices : np.ndarray of ints, shape (n_simplices,)
        Indices of simplices (within `convex_hull.simplices`) forming the facets of the lower convex hull.
    """
    # Find lower hull simplices
    lower_hull_simplex_indices = (-convex_hull.equations[:, -2] > tolerance).nonzero()[
        0
    ]
    if lower_hull_simplex_indices.size == 0:
        raise RuntimeError("No lower hull simplices found.")

    # Gather lower hull vertices from simplices
    lower_hull_vertex_indices = np.unique(
        np.ravel(convex_hull.simplices[lower_hull_simplex_indices])
    )
    return lower_hull_vertex_indices, lower_hull_simplex_indices


def simplex_energy_equation_matrix(
    convex_hull: ConvexHull,
    simplex_indices: Sequence[int],
    tolerance: float = 1e-14,
) -> np.ndarray:
    """Returns a matrix that encodes the energy equation of each requested convex hull simplex.

    Each row in the matrix corresponds to a simplex (as specified by `simplex_indices`).

    Each simplex is described by a hyperplane. For example, in a 3d composition space
    a1*x1 + a2*x2 + a3*x3 + b*e + c = 0
    where x1, x2, x3 are the composition variables and e is the energy.

    Solving for e yields
    e = -(a1*x1 + a2*x2 + a3*x3 + c)/b

    The corresponding row in the equation matrix is then (-a1/b, -a2/b, -a3/b, -c/b),
    such that multiplying with the column vector (x1, x2, x3, 1) yields e.

    Parameters
    ----------
    convex_hull : ConvexHull
        Complete convex hull object. Last coordinate of each point is assumed to be energy.
    simplex_indices : Sequence[int]
        Indices of simplices (within `convex_hull.simplices`) to include in matrix.
    tolerance : float, optional
        Tolerance for checking for vertical hull facets with b close to 0 (default is 1e-14).

    Returns
    -------
    np.ndarray of floats, shape (n_simplex_indices, n_composition_axes + 1)
        Matrix of simplex energy equations.
    """
    # Check for vertical simplices (b = 0 case)
    vertical_simplex_indices = np.array(simplex_indices)[
        (np.abs(convex_hull.equations[simplex_indices, -2]) < tolerance).nonzero()[0]
    ]
    if not vertical_simplex_indices.size == 0:
        raise ValueError(
            f"Vertical hull simplex encountered: Simplex index {','.join(map(str, vertical_simplex_indices))}."
        )

    # Form and return equation matrix
    return (
        -np.delete(convex_hull.equations[simplex_indices, :], -2, axis=1)
        / convex_hull.equations[simplex_indices, -2][:, np.newaxis]
    )


def lower_hull_simplex_containing(
    compositions: np.ndarray,
    convex_hull: ConvexHull,
    lower_hull_simplex_indices: Sequence[int] = None,
    tolerance: float = 1e-14,
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns the lower convex hull simplices of `convex_hull` containing the points in composition space specified by `compositions`, and the corresponding energies.

    For points incident with multiple simplices, one of the simplices is chosen arbitrarily.

    Parameters
    ----------
    compositions : np.ndarray of floats, shape (n_points, n_composition_axes)
        Compositions of points to find containing simplices for. If a 1D array is provided, it is assumed to be a column (multiple points, one composition axis).
    convex_hull : ConvexHull
        Complete convex hull object. Last coordinate of each point is assumed to be energy.
    lower_hull_simplex_indices : Sequence[int], optional
        Indices of lower hull simplices (within `convex_hull.simplices`), if known.
    tolerance : float, optional
        Tolerance for identifying lower hull simplices (default is 1e-14).

    Returns
    -------
    simplex_indices : np.ndarray of ints, shape (n_points,)
        Indices of simplices (within `convex_hull.simplices`) containing each point.
    energies : np.ndarray of floats, shape (n_points,)
        Energy values of points specified by `compositions` on their respective simplices.
    """
    if lower_hull_simplex_indices is None:
        lower_hull_simplex_indices = lower_hull(convex_hull, tolerance=tolerance)[1]

    # Promote 1D composition array to 2D array, if necessary
    if compositions.ndim == 1:
        compositions = compositions[:, np.newaxis]

    # Check that matrices are compatible with one another
    hull_composition_dimension = convex_hull.points.shape[1] - 1
    if not compositions.shape[1] == hull_composition_dimension:
        raise ValueError(
            f"Composition dimensions of input points and hull points differ: {compositions.shape[1]} vs {hull_composition_dimension}."
        )

    # Check composition bounds for input points
    # TODO: Should maybe use lower hull vertices rather than convex_hull.vertices in case of tolerance issues
    out_of_bounds = ~np.array(
        inside_convex_hull(convex_hull.points[convex_hull.vertices, :-1], compositions)
    )
    out_of_bounds_point_indices = out_of_bounds.nonzero()[0]
    if not out_of_bounds_point_indices.size == 0:
        raise ValueError(
            f"Point outside of hull composition bounds encountered: Point index {','.join(map(str, out_of_bounds_point_indices))}."
        )

    # Form equation matrix and multiply with compositions
    lower_hull_equation_matrix = simplex_energy_equation_matrix(
        convex_hull, lower_hull_simplex_indices, tolerance=tolerance
    )
    configuration_simplex_energies = lower_hull_equation_matrix @ np.vstack(
        [compositions.transpose(), np.ones(compositions.shape[0])]
    )

    # Identify and extract correct simplices, energies
    maximum_energy_simplex_indices = np.argmax(configuration_simplex_energies, axis=0)
    simplex_indices = lower_hull_simplex_indices[maximum_energy_simplex_indices]
    energies = np.take_along_axis(
        configuration_simplex_energies,
        maximum_energy_simplex_indices[np.newaxis, :],
        axis=0,
    )[0]
    return simplex_indices, energies


def lower_hull_energies(
    compositions: np.ndarray,
    convex_hull: ConvexHull,
    lower_hull_simplex_indices: Sequence[int] = None,
    tolerance: float = 1e-14,
) -> np.ndarray:
    """Returns energies of points in composition space specified by `compositions` along the lower hull of `convex_hull`.

    Parameters
    ----------
    compositions : np.ndarray of floats, shape (n_points, n_composition_axes)
        Compositions of points to get energies for. If a 1D array is provided, it is assumed to be a column (multiple points, one composition axis).
    convex_hull : ConvexHull
        Complete convex hull object. Last coordinate of each point is assumed to be energy.
    lower_hull_simplex_indices : Sequence[int], optional
        Indices of lower hull simplices (within `convex_hull.simplices`), if known.
    tolerance : float, optional
        Tolerance for identifying lower hull simplices (default is 1e-14).

    Returns
    -------
    np.ndarray of floats, shape (n_points,)
        Energies of points.
    """
    return lower_hull_simplex_containing(
        compositions, convex_hull, lower_hull_simplex_indices, tolerance=tolerance
    )[1]


def lower_hull_distances(
    compositions: np.ndarray,
    energies: np.ndarray,
    convex_hull: ConvexHull = None,
    lower_hull_simplex_indices: Sequence[int] = None,
    tolerance: float = 1e-14,
) -> np.ndarray:
    """Returns hull distances (energy above lower convex hull of `convex_hull`) of points in energy-composition space specified by `compositions` and `energies`.

    If `convex_hull` is omitted, it will be calculated from `compositions` and `energies`.

    Parameters
    ----------
    compositions : np.ndarray of floats, shape (n_points, n_composition_axes)
        Compositions of points to get hull distances for. If a 1D array is provided, it is assumed to be a column (multiple points, one composition axis).
    energies : np.ndarray of floats, shape (n_points,)
        Energies of points to get hull distances for.
    convex_hull : ConvexHull, optional
        Complete convex hull object. Last coordinate of each point is assumed to be energy.
    lower_hull_simplex_indices : Sequence[int], optional
        Indices of lower hull simplices (within `convex_hull.simplices`), if known.
    tolerance : float, optional
        Tolerance for identifying lower hull simplices (default is 1e-14).

    Returns
    -------
    np.ndarray of floats, shape (n_points,)
        Hull distances of points.
    """
    if convex_hull is None:
        convex_hull = full_hull(compositions, energies)
        lower_hull_simplex_indices = None
    return energies - lower_hull_energies(
        compositions, convex_hull, lower_hull_simplex_indices, tolerance=tolerance
    )


def hull_distance_correlations(
    corr: np.ndarray,
    compositions: np.ndarray,
    formation_energy: np.ndarray,
    hull: ConvexHull = None,
) -> np.ndarray:
    """Calculated the effective correlations to predict hull distance instead of absolute formation energy.
    Like formation energies, the hull distance of a point can be described as a scalar product between effective cluster
    interactions (ECI) and some vector of descriptors to characterize the atomic configuration ("hull correlations").
    Assuming that the true ECI were known, the set of ECI could be multiplied with formation energy correlations to
    obtain formation energy predictions; this same set of ECI could be multiplied with the hull correlations to predict
    hull distances.

    The hull correlation of an atomic configuration is found by taking the difference between the correlation for that
    configuration and the linear combination of correlations that define the simplex below the configuration in
    composition-formation_energy space. In this linear combination of ground state correlations, each ground state is
    weighted by the barycentric coordinate in composition space of the configuration of interest.


    Parameters
    ----------
    corr: np.array
        nxk correlation matrix, where n is the number of configurations and k is the number of ECI.
    comp: np.array
        nxc matrix of compositions, where n is the number of configurations and c is the number of composition axes.
    formation_energy: np.array
        nx1 matrix of formation energies.

    Returns
    -------
    hulldist_corr: np.array
        nxk matrix of effective correlations describing hull distance instead of absolute formation energy. n is the number of configurations and k is the number of ECI.
    """

    # Build convex hull from compositions and formation energies
    if hull is None:
        hull = full_hull(compositions=compositions, energies=formation_energy)

    # Get convex hull simplices
    _, lower_simplices = lower_hull(hull)

    hulldist_corr = np.zeros(corr.shape)

    for config_index in list(range(corr.shape[0])):

        # Find the simplex that contains the current configuration's composition, and find the hull energy for that composition
        relevant_simplex_index, _ = lower_hull_simplex_containing(
            compositions=compositions[config_index].reshape(1, -1),
            convex_hull=hull,
            lower_hull_simplex_indices=lower_simplices,
        )

        relevant_simplex_index = relevant_simplex_index[0]

        # Find vectors defining the corners of the simplex which contains the curent configuration's composition.
        simplex_corners = compositions[hull.simplices[relevant_simplex_index]]
        interior_point = np.array(compositions[config_index]).reshape(1, -1)

        # Find barycentric coordinates of the interior point in composition space
        weights = barycentric_coordinates(
            point=interior_point, vertices=simplex_corners
        )

        # Form the hull distance correlations by taking a linear combination of simplex corners.
        hulldist_corr[config_index] = (
            corr[config_index] - weights @ corr[hull.simplices[relevant_simplex_index]]
        )

    return hulldist_corr


def simplex_volume(simplex_vertices: np.ndarray) -> float:
    """Given the vertices of a simplex in n dimensions, calculates the 
    scalar volume analogue of the simplex on the input domain. 
    
    Example: A convex hull in 3D (2 composition dimensions +1 energy dimension) has a simplex that
    is a triangle (2D). For example, take the composition coordinates of the 3 simplex vertices as [[0,0],[1,0],[0,1]]. 
    The area spanned by the simplex vertices in the composition domain is the area of the triangle  [[0,0],[1,0],[0,1]]. 
    equal to 1/2 (which is what this function will produce).  


    Parameters
    ----------
    simplex_vertices: np.ndarray
        nxm matrix of simplex vertices AS ROW VECTORS in n dimensional space. n = m+1
        Note: For CASM applications, n rows are n structures, m columns are m composition dimensions. 

    Returns
    -------
    simplex_volume: float
        The scalar volume-analogue of the given simplex in n dimensions.
    """
    simplex_coords_as_column_matrix = np.hstack(
        (simplex_vertices, np.ones((simplex_vertices.shape[0], 1)))
    ).T
    return np.abs(np.linalg.det(simplex_coords_as_column_matrix)) / factorial(
        simplex_vertices.shape[1]
    )


def trapezoid_rule_n_dim(
    x: np.ndarray, y: np.ndarray, simplices: np.ndarray
) -> np.ndarray:
    """Performs trapezoidal integration in arbitrary dimensions.

    Parameters
    ----------
    x:np.ndarray
        (p,c) matrix of p coordinates in c dimensional input domain space.
    y:np.ndarray
        (p,) array of p scalar outputs of some f(c).
    simplices:np.ndarray
        Matrix of simplices (k,c+1). Each row is a simplex. Each simplex is a vector of (c+1) indices, which index the corresponding input
        x[c] and output y[c]. Simplices must be the same dimension of the full space spanned by c and f(c), meaning a simplex of dimension (c+1).

    Returns
    -------
    integral:float
        Scalar value of the integral.
    """
    integral = 0
    dim_norm = 1 / len(
        simplices[0]
    )  # The dimension of the space, used to get average of y's in a simplex
    for s in simplices:
        integral += simplex_volume(x[s, :]) * np.sum(y[s])
    integral = integral * dim_norm
    return integral


def matching_row_indices(
    rows_of_interest: np.ndarray, rows_to_search: np.ndarray
) -> np.ndarray:
    """Finds indices where a subset of row vectors appear in a larger set of row vectors. Only finds first match, not all matches.
    Parameters
    ----------
    rows_of_interest:np.ndarray
        (n,k) matrix of rows; will search for these rows in the larger set.
    rows_to_search:np.ndarray
        (m,k) matrix of rows, where m >= n, assumed to contain the rows of interest. Returned indices will index into this set.

    Returns
    -------
    matching_indices:np.ndarray
        Vector of (n,) indices (assuming that all rows of "rows_of_interest" can be found in "rows_to_search").

    """
    matching_indices = []
    # Iterate over each row in calc_corr
    for i in range(rows_of_interest.shape[0]):
        # Iterate over each row in uncalc_corr
        for j in range(rows_to_search.shape[0]):
            # If the two rows match
            if np.allclose(rows_of_interest[i], rows_to_search[j]):
                # Record the corresponding index in the uncalculated dataset
                matching_indices.append(j)
                # Break the loop as we found a match
                break
    return np.array(matching_indices)


def index_conversion(true_corr, overenum_corr, only_these_indices=None):
    """Finds where true_corr matches overenum_corr, and creates a dictionary"""

    if any(only_these_indices) == None:
        only_these_indices = list(range(true_corr.shape[0]))

    search_corr = true_corr[only_these_indices, :]

    matching_indices = matching_row_indices(search_corr, overenum_corr)
    return dict(zip(only_these_indices, matching_indices))


def envelope_functional(
    true_hull: ConvexHull, predicted_hull: ConvexHull, index_conversion_dict: dict
):
    """Computes the value of the envelope error: an integral of linear interpolation between known ground states,
    minus integral of predicted convex hull.

    Parameters
    ----------
    true_hull:ConvexHull
        scipy.spatial.ConvexHull object (generated by thermocore.geometry.hull.full_hull) describing the ``true" convex hull.
    predicted_hull:ConvexHull
        scipy.spatial.ConvexHull object (generated by thermocore.geometry.hull.full_hull) describing the predicted convex hull, on a potentially larger
    index_conversion_dict: dict
        Often, the predicted_hull will contain many more points than the true_hull, and the same structure can have 
        a different index in the two data sets. The conversion dictionary takes the index of a structure in the true_hull 
        dataset, and returns the index of that structure in the larger predicted_hull dataset. Generated with the index_conversion 
        function. 
    """
    _, true_simplices = lower_hull(true_hull)
    true_simplices = true_hull.simplices[true_simplices]

    # The error functional needs to reference the true ground state structures within the predicted data set.
    # Indices don't necessarily match. Find the corresponding indices in the predicted set using provided index conversion dictionary
    translated_true_simplices = []
    for simplex in true_simplices:
        translated_true_simplices.append([index_conversion_dict[s] for s in simplex])

    predicted_x = predicted_hull.points[:, 0:-1]
    predicted_y = predicted_hull.points[:, -1]
    _, predicted_simplices = lower_hull(predicted_hull)
    predicted_simplices = predicted_hull.simplices[predicted_simplices]

    return trapezoid_rule_n_dim(
        predicted_x, predicted_y, translated_true_simplices
    ) - trapezoid_rule_n_dim(predicted_x, predicted_y, predicted_simplices)


def envelope_functional_gradient(
    corr_calc,
    comp_calc,
    eng_calc,
    corr_overenum,
    comp_overenum,
    eng_predicted,
    calc_hull: ConvexHull = None,
    predicted_hull: ConvexHull = None,
):
    """Gradient of the cone error functional, abstracted to any composition dimension.
    Parameters
    ----------
    corr_calc: np.ndarray
        Array of correlations for the calculated dataset. Shape (n,k) of n configurations, k eci. 
    comp_calc: np.ndarray
        Array of compositions for calculated configurations. Shape (n,m) of n configurations, m composition dimensions. 
    eng_calc: np.ndarray
        Array of calculated formation energies. Shape (n,) of n configurations. 
    corr_overenum: np.ndarray
        Array of over-enumerated correlations, containing the entirety of corr_calc. Shape (p,k), p>n, of p configurations, k eci. 
    comp_overenum: np.ndarray
        Array of compositions for over-enumerated configurations. Shape (p,m) of p configurations, m composition dimensions. 
    eng_predicted: np.ndarray
        Array of model-predicted formation energies. Shape (p,) 
    calc_hull: ConvexHull
        Optional convex hull of calculated configurations. Saves time if provided. 
    predicted_hull: ConvexHull 
        Optional convex hull of predicted configurations. Saves time if provided.  
    
    Returns
    -------
    negative_gradient: np.ndarray
        Negative gradient of the hull envelope error functional. Points in the direction that minimizes 
        the envelope error functioinal. Shape (k,) of k eci dimensions. 
    """
    # Calculate hulls if not provided
    if calc_hull == None:
        calc_hull = full_hull(comp_calc, eng_calc)
    if predicted_hull == None:
        predicted_hull = full_hull(comp_overenum, eng_predicted)

    # Calculate relevant hull quantities
    _, calc_simplices = lower_hull(calc_hull)
    calc_simplices = calc_hull.simplices[calc_simplices]
    _, predicted_simplices = lower_hull(predicted_hull)
    predicted_simplices = predicted_hull.simplices[predicted_simplices]

    # Calculate individual contributions to the gradient
    calc_grad = np.zeros(corr_calc.shape[1])
    predicted_grad = np.zeros(corr_calc.shape[1])
    for simplex in calc_simplices:
        calc_grad += np.sum(corr_calc[simplex], axis=0) * simplex_volume(
            comp_calc[simplex, :]
        )
    for simplex in predicted_simplices:
        predicted_grad += np.sum(corr_overenum[simplex], axis=0) * simplex_volume(
            comp_overenum[simplex, :]
        )

    return (predicted_grad - calc_grad) * (1 / simplex.shape[0])
