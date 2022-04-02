#!/usr/bin/env python

from concurrent.futures import ProcessPoolExecutor
from time import time
from functools import wraps
import numpy as np
import numpy.linalg as linalg
import pandas as pd
import sklearn as sklearn
from sklearn.neighbors import KDTree
import math

from tqdm import tqdm
import random

import cupy as cp
import cudf
from numba import cuda

executor = ProcessPoolExecutor(4)


def timeer(func):
    @wraps(func)
    def wrap(*args, **kw):
        start = time()
        result = func(*args, **kw)
        end = time()
        print(f"Timer,{func.__name__},{(end-start)}")
        return result
    return wrap

# covariance function definition


def covar(t, d, r):
    h = d / r
    if t == 1:  # Spherical
        c = 1 - h * (1.5 - 0.5 * cp.square(h))
        c[h > 1] = 0
    elif t == 2:  # Exponential
        c = cp.exp(-3 * h)
    elif t == 3:  # Gaussian
        c = cp.exp(-3 * cp.square(h))
    return c


def formatQuadrant(q, loc):
    # calculate distances of points from location
    dist = cp.linalg.norm(q - loc, axis=1)
    # transpose vector to add it to our data matrix
    dist = dist.reshape(len(dist), 1)
    return cp.append(q, dist, axis=1)


def sortQuadrantPoints(quad_array, quad_count, rad, loc):
    quad = quad_array[:, :2]
    cp.asarray(quad)
    quad = formatQuadrant(quad, loc)
    quad = cp.insert(quad, 3, quad_array[:, -1], axis=1)
    # sort array by smallest distance values
    quad = quad[cp.argsort(quad[:, 2])]
    # select the number of points in each quadrant up to our quadrant count
    smallest = quad[:quad_count]
    # delete points outside of our radius
    smallest = cp.delete(smallest, cp.where((smallest[:, 2] > rad))[0], 0)
    # delete extra column of distance from origin data
    smallest = cp.delete(smallest, 2, 1)
    return smallest


def nearestNeighborSearch(rad, count, loc, data):
    locx = loc[0]
    locy = loc[1]

    # wipe coords for re-usability

    # coords = cp.copy(data)
    coords = cp.asarray(data)
    print(type(coords), type(locx), type(locy))
    # standardize our quadrants (create the origin at our location point)
    coords[:, 0] -= locx
    coords[:, 1] -= locy

    # Number of points to look for in each quadrant, if not fully divisible by 4, round down
    quad_count = count//4

    # sort coords of dataset into 4 quadrants relative to input location
    final_quad = []
    final_quad.append(coords[(coords[:, 0] >= 0) & (coords[:, 1] >= 0)])
    final_quad.append(coords[(coords[:, 0] < 0) & (coords[:, 1] < 0)])
    final_quad.append(coords[(coords[:, 0] >= 0) & (coords[:, 1] < 0)])
    final_quad.append(coords[(coords[:, 0] < 0) & (coords[:, 1] >= 0)])

    # Gather distance values for each coord from point and delete points outside radius
    # Use executor to submit function
    final_fs = [sortQuadrantPoints(fquad, quad_count, rad, cp.array((0, 0))) for fquad in final_quad]
    ### fs = [executor.submit(sortQuadrantPoints, fquad, quad_count, rad, cp.array((0, 0))) for fquad in final_quad]

    # Get results back
    ### final_fs = [f.result() for f in fs]

    # add all quadrants back together for final dataset
    # TODO: Hstack or vstack????
    near = cp.concatenate((final_fs[0], final_fs[1]))
    near = cp.concatenate((near, final_fs[2]))
    near = cp.concatenate((near, final_fs[3]))
    # unstandardize data back to original form
    near[:, 0] += locx
    near[:, 1] += locy
    return near


# get variogram along the major or minor axis
def axis_var(lagh, nug, nstruct, cc, vtype, a):
    lagh = lagh
    nstruct = nstruct  # number of variogram structures
    vtype = vtype  # variogram types (Gaussian, etc.)
    a = a  # range for axis in question
    cc = cc  # contribution of each structure

    n = len(lagh)
    gamma_model = cp.zeros(shape=(n))

    # for each lag distance
    for j in range(0, n):
        c = nug
        c = 0
        h = cp.matrix(lagh[j])

        # for each structure in the variogram
        for i in range(nstruct):
            Q = h.copy()
            d = Q / a[i]
            c = c + covar(vtype[i], d, 1) * cc[i]  # covariance

        gamma_model[j] = 1 + nug - c  # variance
    return gamma_model


# make array of x,y coordinates based on corners and resolution
@timeer
def pred_grid(xmin, xmax, ymin, ymax, pix):
    cols = cp.rint((xmax - xmin)/pix)
    rows = cp.rint((ymax - ymin)/pix)  # number of rows and columns
    x = cp.arange(xmin, xmax, pix)
    y = cp.arange(ymin, ymax, pix)  # make arrays

    xx, yy = cp.meshgrid(x, y)  # make grid
    yy = cp.flip(yy)  # flip upside down

    # shape into array
    x = cp.reshape(xx, (int(rows)*int(cols), 1))
    y = cp.reshape(yy, (int(rows)*int(cols), 1))

    Pred_grid_xy = cp.concatenate((x, y), axis=1)  # combine coordinates
    return Pred_grid_xy


# rotation matrix (Azimuth = major axis direction)
def Rot_Mat(Azimuth, a_max, a_min):
    theta = (Azimuth / 180.0) * cp.pi
    Rot_Mat = cp.dot(
        cp.array([[1 / a_max, 0], [0, 1 / a_min]]),
        cp.array(
            [
                [cp.cos(theta), cp.sin(theta)],
                [-cp.sin(theta), cp.cos(theta)],
            ]
        ),
    )
    return Rot_Mat


def minkowski_distance_p(x, y, p=2):
    """Compute the pth power of the L**p distance between two arrays.
    For efficiency, this function computes the L**p distance but does
    not extract the pth root. If `p` is 1 or infinity, this is equal to
    the actual L**p distance.
    Parameters
    ----------
    x : (M, K) array_like
        Input array.
    y : (N, K) array_like
        Input array.
    p : float, 1 <= p <= infinity
        Which Minkowski p-norm to use.
    Examples
    --------
    >>> from scipy.spatial import minkowski_distance_p
    >>> minkowski_distance_p([[0,0],[0,0]], [[1,1],[0,1]])
    array([2, 1])
    """
    x = cp.asarray(x)
    y = cp.asarray(y)

    if p == cp.inf:
        return cp.amax(cp.abs(y-x), axis=-1)
    elif p == 1:
        return cp.sum(cp.abs(y-x), axis=-1)
    else:
        return cp.sum(cp.abs(y-x)**p, axis=-1)


def minkowski_distance(x, y, p=2):
    """Compute the L**p distance between two arrays.
    Parameters
    ----------
    x : (M, K) array_like
        Input array.
    y : (N, K) array_like
        Input array.
    p : float, 1 <= p <= infinity
        Which Minkowski p-norm to use.
    Examples
    --------
    >>> from scipy.spatial import minkowski_distance
    >>> minkowski_distance([[0,0],[0,0]], [[1,1],[0,1]])
    array([ 1.41421356,  1.        ])
    """
    x = cp.asarray(x)
    y = cp.asarray(y)
    if p == cp.inf or p == 1:
        return minkowski_distance_p(x, y, p)
    else:
        return minkowski_distance_p(x, y, p)**(1./p)


def distance_matrix(x, y, p=2, threshold=1000000):
    """Compute the distance matrix.
    Returns the matrix of all pair-wise distances.
    Parameters
    ----------
    x : (M, K) array_like
        Matrix of M vectors in K dimensions.
    y : (N, K) array_like
        Matrix of N vectors in K dimensions.
    p : float, 1 <= p <= infinity
        Which Minkowski p-norm to use.
    threshold : positive int
        If ``M * N * K`` > `threshold`, algorithm uses a Python loop instead
        of large temporary arrays.
    Returns
    -------
    result : (M, N) ndarray
        Matrix containing the distance from every vector in `x` to every vector
        in `y`.
    Examples
    --------
    >>> from scipy.spatial import distance_matrix
    >>> distance_matrix([[0,0],[0,1]], [[1,0],[1,1]])
    array([[ 1.        ,  1.41421356],
           [ 1.41421356,  1.        ]])
    """

    x = cp.asarray(x)
    m, k = x.shape
    y = cp.asarray(y)
    n, kk = y.shape

    if k != kk:
        raise ValueError("x contains %d-dimensional vectors but y contains %d-dimensional vectors" % (k, kk))

    if m*n*k <= threshold:
        result = minkowski_distance(x[:, cp.newaxis, :], y[cp.newaxis, :, :], p)
    else:
        result = cp.empty((m, n), dtype=cp.float32)  # FIXME: figure out the best dtype
        if m < n:
            for i in range(m):
                result[i, :] = minkowski_distance(x[i], y, p)
        else:
            for j in range(n):
                result[:, j] = minkowski_distance(x, y[j], p)

    return cp.asnumpy(result)

# covariance model


def cov(h1, h2, k, vario):
    # unpack variogram parameters
    Azimuth = vario[0]
    nug = vario[1]
    nstruct = vario[2]
    vtype = vario[3]
    cc = vario[4]
    a_max = vario[5]
    a_min = vario[6]

    c = -nug  # nugget effect is made negative because we're calculating covariance instead of variance
    for i in range(nstruct):
        Q1 = h1.copy()
        Q2 = h2.copy()

        # covariances between measurements
        if k == 0:
            d = distance_matrix(
                cp.matmul(Q1, Rot_Mat(Azimuth, a_max[i], a_min[i])),
                cp.matmul(Q2, Rot_Mat(Azimuth, a_max[i], a_min[i])),
            )

        # covariances between measurements and unknown
        elif k == 1:
            d = cp.sqrt(
                cp.square(
                    (cp.matmul(Q1, Rot_Mat(Azimuth, a_max[i], a_min[i])))
                    - cp.tile(
                        (
                            cp.matmul(
                                Q2, Rot_Mat(Azimuth, a_max[i], a_min[i])
                            )
                        ),
                        (k, 1),
                    )
                ).sum(axis=1)
            )
            d = cp.asarray(d).reshape(len(d))
        c = c + covar(vtype[i], d, 1) * cc[i]
    return c


######################################

# Simple Kriging Function

######################################


def skrige(Pred_grid, df, xx, yy, data, k, vario, rad):
    """Simple kriging interpolation
    :param Pred_grid: x,y coordinate numpy array of prediction grid
    :df: data frame of input data
    :xx: column name for x coordinates of input data frame
    :yy: column name for y coordinates of input data frame
    :data: column name for the data variable (in this case, bed elevation) of the input data frame
    :k: the number of conditioning points to search for
    :vario: variogram parameters describing the spatial statistics
    """

    Mean_1 = cp.average(df[data])  # mean of input data
    Var_1 = cp.var(df[data])  # variance of input data

    # preallocate space for mean and variance
    est_SK = cp.zeros(shape=len(Pred_grid))  # make zeros array the size of the prediction grid
    var_SK = cp.zeros(shape=len(Pred_grid))

    # convert dataframe to numpy array for faster matrix operations
    npdata = df[['X', 'Y', 'Nbed']].to_numpy()
    npdata = cp.asarray(npdata)

    # for each coordinate in the prediction grid
    for z in tqdm(range(0, len(Pred_grid))):
        # gather nearest points within radius
        nearest = nearestNeighborSearch(rad, k, Pred_grid[z], npdata)

        # format nearest point bed values matrix
        norm_bed_val = nearest[:, -1]
        norm_bed_val = norm_bed_val.reshape(len(norm_bed_val), 1)
        norm_bed_val = norm_bed_val.T
        xy_val = nearest[:, :-1]

        # calculate new_k value relative to count of near points within radius
        new_k = len(nearest)
        Kriging_Matrix = cp.zeros(shape=((new_k, new_k)))
        Kriging_Matrix = cov(xy_val, xy_val, 0, vario)

        r = cp.zeros(shape=(new_k))
        k_weights = r
        r = cov(xy_val, cp.tile(Pred_grid[z], (new_k, 1)), 1, vario)
        Kriging_Matrix.reshape(((new_k)), ((new_k)))

        k_weights = cp.dot(cp.linalg.pinv(Kriging_Matrix), r)
        est_SK[z] = new_k*Mean_1 + (cp.sum(k_weights*(norm_bed_val[:] - Mean_1)))
        var_SK[z] = (Var_1 - cp.sum(k_weights*r))

    return est_SK, var_SK


###########################

# Ordinary Kriging Function

###########################

def okrige(Pred_grid, df, xx, yy, data, k, vario, rad):
    """Ordinary kriging interpolation
    :param Pred_grid: x,y coordinate numpy array of prediction grid
    :df: data frame of input data
    :xx: column name for x coordinates of input data frame
    :yy: column name for y coordinates of input data frame
    :data: column name for the data variable (in this case, bed elevation) of the input data frame
    :k: the number of conditioning points to search for
    :vario: variogram parameters describing the spatial statistics
    """

    Var_1 = cp.var(df[data])  # variance of data

    # preallocate space for mean and variance
    est_OK = cp.zeros(shape=len(Pred_grid))
    var_OK = cp.zeros(shape=len(Pred_grid))

    # convert dataframe to numpy matrix for faster operations
    npdata = df[['X', 'Y', 'Nbed']].to_numpy()

    for z in tqdm(range(0, len(Pred_grid))):
        # find nearest data points
        nearest = nearestNeighborSearch(rad, k, Pred_grid[z], npdata)

        # format matrix of nearest bed values
        norm_bed_val = nearest[:, -1]
        norm_bed_val = norm_bed_val.reshape(len(norm_bed_val), 1)
        norm_bed_val = norm_bed_val.T
        xy_val = nearest[:, :-1]

        # calculate new_k value relative to number of nearby points within radius
        new_k = len(nearest)

        # left hand side (covariance between data)
        Kriging_Matrix = cp.zeros(shape=((new_k+1, new_k+1)))
        Kriging_Matrix[0:new_k, 0:new_k] = cov(xy_val, xy_val, 0, vario)
        Kriging_Matrix[new_k, 0:new_k] = 1
        Kriging_Matrix[0:new_k, new_k] = 1

        # Set up Right Hand Side (covariance between data and unknown)
        r = cp.zeros(shape=(new_k+1))
        k_weights = r
        r[0:new_k] = cov(xy_val, cp.tile(Pred_grid[z], (new_k, 1)), 1, vario)
        r[new_k] = 1  # unbiasedness constraint
        Kriging_Matrix.reshape(((new_k+1)), ((new_k+1)))

        # Calculate Kriging Weights
        k_weights = cp.dot(cp.linalg.pinv(Kriging_Matrix), r)

        # get estimates
        est_OK[z] = cp.sum(k_weights[0:new_k]*norm_bed_val[:])
        var_OK[z] = Var_1 - cp.sum(k_weights[0:new_k]*r[0:new_k])

    return est_OK, var_OK


# sequential Gaussian simulation
def sgsim(Pred_grid, df, xx, yy, data, k, vario, rad):
    """Sequential Gaussian simulation
    :param Pred_grid: x,y coordinate numpy array of prediction grid
    :df: data frame of input data
    :xx: column name for x coordinates of input data frame
    :yy: column name for y coordinates of input data frame
    :df: column name for the data variable (in this case, bed elevation) of the input data frame
    :k: the number of conditioning points to search for
    :vario: variogram parameters describing the spatial statistics
    """

    # generate random array for simulation order
    xyindex = cp.arange(len(Pred_grid))
    random.shuffle(xyindex)

    Var_1 = cp.var(df[data])  # variance of data

    # preallocate space for simulation
    sgs = cp.zeros(shape=len(Pred_grid))

    for i in tqdm(range(0, len(Pred_grid)), position=0, leave=True):
        z = xyindex[i]

        # convert data to numpy array for faster speeds/parsing
        npdata = df[['X', 'Y', 'Nbed']].to_numpy()

        # gather nearby points
        nearest = nearestNeighborSearch(rad, k, Pred_grid[z], npdata)

        # store bed elevation values in new array
        norm_bed_val = nearest[:, -1]
        norm_bed_val = norm_bed_val.reshape(-1, 1)
        norm_bed_val = norm_bed_val.T
        # store X,Y pair values in new array
        xy_val = nearest[:, :-1]

        # update K to reflect the amount of K values we got back from quadrant search
        new_k = len(nearest)

        # left hand side (covariance between data)
        Kriging_Matrix = cp.zeros(shape=((new_k+1, new_k+1)))
        Kriging_Matrix[0:new_k, 0:new_k] = cov(xy_val, xy_val, 0, vario)
        Kriging_Matrix[new_k, 0:new_k] = 1
        Kriging_Matrix[0:new_k, new_k] = 1

        # Set up Right Hand Side (covariance between data and unknown)
        r = cp.zeros(shape=(new_k+1))
        k_weights = r
        r[0:new_k] = cov(xy_val, cp.tile(Pred_grid[z], (new_k, 1)), 1, vario)
        r[new_k] = 1  # unbiasedness constraint
        Kriging_Matrix.reshape(((new_k+1)), ((new_k+1)))

        # Calculate Kriging Weights
        k_weights = cp.dot(cp.linalg.pinv(Kriging_Matrix), r)

        # get estimates
        est = cp.sum(k_weights[0:new_k]*norm_bed_val[:])  # kriging mean
        var = Var_1 - cp.sum(k_weights[0:new_k]*r[0:new_k])  # kriging variance

        if (var < 0):  # make sure variances are non-negative
            var = 0

        sgs[z] = cp.random.normal(est, math.sqrt(var), 1)  # simulate by randomly sampling a value

        # update the conditioning data
        coords = Pred_grid[z:z+1, :]
        dnew = {xx: [coords[0, 0]], yy: [coords[0, 1]], data: [sgs[z]]}
        dfnew = pd.DataFrame(data=dnew)
        df = pd.concat([df, dfnew], sort=False)  # add new points by concatenating dataframes

    return sgs
