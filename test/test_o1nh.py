from time import sleep
import unittest
import numpy as np

from O1NumHess import O1NumHess


def f(x: np.ndarray) -> float:
    r"""
    $$f(\mathbf{x})=\sum_{i=1}^n a_i x_i+\sin\left(\sum_{i=1}^nb_ix_i\right)$$

    $a_i = i$, $b_i = n-i$
    """
    n = len(x)
    a = np.arange(1, n + 1)  # a_i = i
    b = np.arange(n, 0, -1)  # b_i = n - i
    res = np.sum(a * x) + np.sin(np.sum(b * x))

    return res


def grad_f(x: np.ndarray, index: int, core=0, **kwargs) -> np.ndarray:
    r"""
    $$g(x_i)=\frac{\partial f}{\partial x_i} = a_i + b_i \cos\left( \sum_{j=1}^n b_j x_j \right)$$

    $a_i = i$, $b_i = n-i$
    """
    n = len(x)
    a = np.arange(1, n + 1)  # a_i = i
    b = np.arange(n, 0, -1)  # b_i = n - i
    res = a + b * np.cos(np.dot(b, x))

    # use sleep to imitate parallel running
    if core > 0:
        t = np.random.rand()
        print(f"grad func begin to sleep {t} seconds, {kwargs}")
        sleep(t)
        print(f"grad func wake up from sleep {t} seconds, {kwargs}")

    return res


class TestO1NumHess(unittest.TestCase):
    def setUp(self):
        self.x = np.array([ 1.13854015, -9.72379374, -2.05165172,  9.7018372 ,  6.06292507])
        self.o1nh = O1NumHess(self.x, grad_func=grad_f)
        print("\n" + "="*50)
        print(f"Starting test: {self._testMethodName}")
        print("="*50)

    def tearDown(self):
        del self.o1nh
        print("-"*50)
        print(f"Finished test: {self._testMethodName}")
        print("-"*50 + "\n")

    def test_parallel_vs_serial(self):
        """Verify that parallel and serial computations produce the same result"""
        hessian_single = self.o1nh.singleSide(core=4)
        _hessian_single = self.o1nh._singleSide()
        # print(hessian_single)
        np.testing.assert_array_almost_equal(
            hessian_single, _hessian_single, decimal=6,
            err_msg="Parallel and serial singleSide results differ"
        )

        hessian_double = self.o1nh.doubleSide(core=4)
        _hessian_double = self.o1nh._doubleSide()
        # print(hessian_double)
        np.testing.assert_array_almost_equal(
            hessian_double, _hessian_double, decimal=6,
            err_msg="Parallel and serial doubleSide results differ"
        )

    def test_invalid_core_numbers(self):
        """Test handling of invalid CPU core counts"""
        # Odd core count 5 should still work
        with self.assertWarns(RuntimeWarning):
            self.assertEqual(self.o1nh.singleSide(core=5).shape, (5,5))

        # core=0 should default to 1
        self.assertEqual(self.o1nh.singleSide(core=0).shape, (5,5))

        # Negative cores should raise ValueError
        with self.assertRaises(ValueError):
            self.o1nh.singleSide(core=-1)

        # Excessively large core count should raise ValueError
        with self.assertRaises(ValueError):
            self.o1nh.singleSide(core=999)

    def test_total_cores(self):
        """Test handling of different total_cores

        Can not test the case: os.cpu_count() is None
        """
        with self.assertWarns(RuntimeWarning):
            self.o1nh.doubleSide(total_cores=None)
        with self.assertRaises(ValueError):
            self.o1nh.doubleSide(total_cores=999)
        with self.assertRaises(TypeError):
            self.o1nh.doubleSide(total_cores=1.2) # type: ignore

    def test_different_data_types(self):
        """Test various input data types"""
        test_cases = [
            ("Constant", 1, False),
            ("1D list", [1], True),
            ("2D single-row list", [[1, 2]], True),
            ("2D single-row ndarray", np.array([[1, 2]]), True),
            ("Matrix", np.matrix([[1, 2]]), True),
        ]

        for name, x, should_pass in test_cases:
            if should_pass:
                o1nh = O1NumHess(x, grad_f)
                hessian = o1nh.singleSide()
                self.assertTrue(isinstance(hessian, np.ndarray))
            else:
                with self.assertRaises(Exception):
                    O1NumHess(x, grad_f)

    def test_invalid_dimensions(self):
        """Test invalid input dimensions"""
        invalid_shapes = [
            np.array([[1], [2]]),    # Column vector
            np.array([[[1]]]),       # 3D with single element
            np.array([[[1, 2]]])     # 3D with row vector
        ]

        for x in invalid_shapes:
            with self.assertRaises((ValueError, AssertionError)):
                O1NumHess(x, grad_f)

if __name__ == "__main__":
    unittest.main()
