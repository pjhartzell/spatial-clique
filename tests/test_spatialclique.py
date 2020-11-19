import pytest
import numpy as np
from spatialclique import hard_adjacency, soft_adjacency


@pytest.fixture
def src():
    """Simple spatial graph nodes"""
    src = np.array([[0,0],
                    [1,0],
                    [0,1],
                    [1,1]])
    return src

@pytest.fixture
def dst():
    """Simple spatial graph nodes where all but one have been translated"""
    dst = np.array([[0,0],
                    [1,0.1],
                    [0,1.1],
                    [1,1.1]])
    return dst

@pytest.fixture
def src_cov():
    """Zero (for simplicity) source covariance matrices"""
    s = 0.0
    src_cov = np.array([
        [[s**2, 0],
         [0, s**2]],
        [[s**2, 0],
         [0, s**2]],
        [[s**2, 0],
         [0, s**2]],
        [[s**2, 0],
         [0, s**2]]
    ])
    return src_cov

@pytest.fixture
def dst_cov_small():
    """Destination covariance matrices that will result in some edges
    connected to the unmoved node being excluded from the adjacency matrix"""
    s = 0.05
    dst_cov = np.array([
        [[s**2, 0],
         [0, s**2]],
        [[s**2, 0],
         [0, s**2]],
        [[s**2, 0],
         [0, s**2]],
        [[s**2, 0],
         [0, s**2]]
    ])
    return dst_cov

@pytest.fixture
def dst_cov_large():
    """Destination covariance matrices that will result in all node edges
    being included in the adjacency matrix"""
    s = 0.08
    dst_cov = np.array([
        [[s**2, 0],
         [0, s**2]],
        [[s**2, 0],
         [0, s**2]],
        [[s**2, 0],
         [0, s**2]],
        [[s**2, 0],
         [0, s**2]]
    ])
    return dst_cov


def test_hard_adjacency_small(src, dst):
    """Do we produce a correct adjacency matrix with a hard threshold that
    should eliminate some edges"""
    threshold = 0.07
    test_adj = hard_adjacency(src, dst, threshold)
    known_adj = np.array(
        [
            [1, 1, 0, 0],
            [1, 1, 1, 1],
            [0, 1, 1, 1],
            [0, 1, 1, 1]
        ]
    )
    assert (test_adj == known_adj).all()

def test_hard_adjacency_large(src, dst):
    """Do we produce a correct adjacency matrix with a hard threshold that
    should allow all edges"""
    threshold = 0.11
    test_adj = hard_adjacency(src, dst, threshold)
    known_adj = np.array(
        [
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1]
        ]
    )
    assert (test_adj == known_adj).all()

def test_soft_adjacency_small(src, dst, src_cov, dst_cov_small):
    """Do we produce a correct adjacency matrix with a soft threshold that
    should eliminate some edges"""
    confidence = 68.2689492
    test_adj = soft_adjacency(src, dst, src_cov, dst_cov_small, confidence)
    known_adj = np.array(
        [
            [1, 1, 0, 0],
            [1, 1, 1, 1],
            [0, 1, 1, 1],
            [0, 1, 1, 1]
        ]
    )
    assert (test_adj == known_adj).all()

def test_soft_adjacency_large(src, dst, src_cov, dst_cov_large):
    """Do we produce a correct adjacency matrix with a soft threshold that
    should allow all edges"""
    confidence = 68.2689492
    test_adj = soft_adjacency(src, dst, src_cov, dst_cov_large, confidence)
    known_adj = np.array(
        [
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1]
        ]
    )
    assert (test_adj == known_adj).all()
