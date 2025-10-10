import numpy as np
import pytest
from src.linear_thompson import LinearThompson

def test_initialization():
    num_arms = 3
    dim = 5
    lt = LinearThompson(num_arms, dim)
    assert lt.num_arms == num_arms
    assert lt.dim == dim
    assert len(lt.A) == num_arms
    assert lt.A[0].shape == (dim, dim)
    assert len(lt.b) == num_arms
    assert lt.b[0].shape == (dim,)
    assert len(lt.invA) == num_arms
    assert lt.invA[0].shape == (dim, dim)

def test_sample_arm():
    num_arms = 2
    dim = 3
    lt = LinearThompson(num_arms, dim, noise_var=0.1)
    x = np.array([1.0, 2.0, 3.0])
    arm = lt.sample_arm(x)
    assert 0 <= arm < num_arms

def test_update():
    num_arms = 1
    dim = 2
    lt = LinearThompson(num_arms, dim, lambda_prior=1.0, noise_var=1.0)
    arm = 0
    x = np.array([1.0, 0.5])
    reward = 10.0

    # Store initial values
    initial_A = lt.A[arm].copy()
    initial_b = lt.b[arm].copy()
    initial_invA = lt.invA[arm].copy()

    lt.update(arm, x, reward)

    # Check if A and b are updated
    assert not np.array_equal(lt.A[arm], initial_A)
    assert not np.array_equal(lt.b[arm], initial_b)
    assert not np.array_equal(lt.invA[arm], initial_invA)

    # Verify the update logic for A and b
    expected_A = initial_A + np.outer(x, x) / lt.noise_var
    expected_b = initial_b + x * (reward / lt.noise_var)
    np.testing.assert_allclose(lt.A[arm], expected_A)
    np.testing.assert_allclose(lt.b[arm], expected_b)

    # Verify invA is the inverse of the updated A
    np.testing.assert_allclose(lt.invA[arm], np.linalg.inv(lt.A[arm]))

def test_get_and_from_state():
    num_arms = 2
    dim = 3
    lt = LinearThompson(num_arms, dim)
    x1 = np.array([1.0, 0.0, 0.0])
    x2 = np.array([0.0, 1.0, 0.0])
    lt.update(0, x1, 5.0)
    lt.update(1, x2, 7.0)

    state = lt.get_state()
    new_lt = LinearThompson.from_state(state)

    assert new_lt.num_arms == lt.num_arms
    assert new_lt.dim == lt.dim
    assert new_lt.lambda_prior == lt.lambda_prior
    assert new_lt.noise_var == lt.noise_var
    for i in range(num_arms):
        np.testing.assert_allclose(new_lt.A[i], lt.A[i])
        np.testing.assert_allclose(new_lt.b[i], lt.b[i])
        np.testing.assert_allclose(new_lt.invA[i], lt.invA[i])

def test_numeric_fallback_sample_arm(mocker):
    num_arms = 1
    dim = 2
    lt = LinearThompson(num_arms, dim)
    x = np.array([1.0, 1.0])

    # Mock multivariate_normal to raise an exception
    mocker.patch('numpy.random.multivariate_normal', side_effect=np.linalg.LinAlgError)

    arm = lt.sample_arm(x)
    assert arm == 0

def test_numeric_fallback_update(mocker):
    num_arms = 1
    dim = 2
    lt = LinearThompson(num_arms, dim)
    arm = 0
    x = np.array([1.0, 1.0])
    reward = 1.0

    # Corrupt A to make it singular
    lt.A[arm] = np.zeros((dim, dim))

    # Mock np.linalg.inv to raise an exception
    mocker.patch('numpy.linalg.inv', side_effect=np.linalg.LinAlgError)

    lt.update(arm, x, reward)
    # The fallback should have been triggered, and invA should be updated with regularization
    assert not np.array_equal(lt.invA[arm], np.zeros((dim, dim)))
    # Further checks could involve verifying the regularization term if needed

def test_initialization_validation():
    with pytest.raises(ValueError, match="num_arms must be a positive integer"):
        LinearThompson(num_arms=0, dim=1)
    with pytest.raises(ValueError, match="num_arms must be a positive integer"):
        LinearThompson(num_arms=-1, dim=1)
    with pytest.raises(ValueError, match="num_arms must be a positive integer"):
        LinearThompson(num_arms=1.5, dim=1)
    with pytest.raises(ValueError, match="dim must be a positive integer"):
        LinearThompson(num_arms=1, dim=0)
    with pytest.raises(ValueError, match="dim must be a positive integer"):
        LinearThompson(num_arms=1, dim=-1)
    with pytest.raises(ValueError, match="dim must be a positive integer"):
        LinearThompson(num_arms=1, dim=2.5)
    with pytest.raises(ValueError, match="lambda_prior must be a positive float"):
        LinearThompson(num_arms=1, dim=1, lambda_prior=0)
    with pytest.raises(ValueError, match="lambda_prior must be a positive float"):
        LinearThompson(num_arms=1, dim=1, lambda_prior=-1.0)
    with pytest.raises(ValueError, match="noise_var must be a positive float"):
        LinearThompson(num_arms=1, dim=1, noise_var=0)
    with pytest.raises(ValueError, match="noise_var must be a positive float"):
        LinearThompson(num_arms=1, dim=1, noise_var=-1.0)
