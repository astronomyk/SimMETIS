'''Unit tests for module simmetis.utils'''

import pytest
import simmetis as sim
from simmetis.utils import parallactic_angle, deriv_polynomial2d
from simmetis.utils import find_file
import numpy as np


class TestFindFile():
    '''Tests of function simmetis.utils.find_file'''
    search_path = ['./', sim.__pkg_dir__]
    def test_01(self):
        '''Test: fail if not a string'''
        with pytest.raises(TypeError):
            find_file(1.2, self.search_path)

    def test_02(self):
        '''Test: existing file'''
        filename = 'utils.py'
        assert find_file(filename, self.search_path)

    def test_03(self):
        '''Test: non-extisting file'''
        filename = 'utils987654.pz'
        assert find_file(filename, self.search_path) is None


class TestParallacticAngle():
    '''Tests of function simmetis.utils.parallactic_angle'''

    def test_01(self):
        '''Test: parallactic angle negative east of meridian'''
        assert parallactic_angle(-1, 0, -24) < 0

    def test_02(self):
        '''Test: parallactic angle positive west of meridian'''
        assert parallactic_angle(1, 0, -24) > 0

    def test_03(self):
        '''Test: parallactic angle zero on meridian'''
        assert parallactic_angle(0, 0, 24) == 0

    def test_04(self):
        '''Test: Example from Ball (1908), p.92'''
        ha = -3                 # 3 hours east
        de = 38 + 9/60          # decl 38d09m
        lat = 53 + 23/60        # lat  53d23m
        eta0 = - (48 + 41/60)   # result -48d41m

        eta = parallactic_angle(ha, de, lat)

        # should agree to within 1 arcmin
        assert np.allclose(eta, eta0, atol=1/60)

    def test_05(self):
        '''Test parallactic angle

        For a setting object on the equator, the parallactic angle is 90 - lat'''
        lat = np.random.rand(10) * 180 - 90
        pa = parallactic_angle(6, 0, lat)

        assert np.allclose(pa, 90. - lat)


class TestDerivPolynomial2D():
    '''Tests of simmetis.utils.deriv_polynomial2d'''

    def test_01(self):
        '''Test simmetis.utils.deriv_polynomial2d'''
        from astropy.modeling.models import Polynomial2D

        ximg, yimg = np.meshgrid(np.linspace(-1, 1, 101),
                                 np.linspace(-1, 1, 101))
        poly = Polynomial2D(2, c0_0=1, c1_0=2, c2_0=3,
                            c0_1=-1.5, c0_2=0.4, c1_1=-2)
        # Expected values
        y_x = 2 + 6 * ximg - 2 * yimg
        y_y = -1.5 + 0.8 * yimg - 2 * ximg

        dpoly_x, dpoly_y = deriv_polynomial2d(poly)
        # Computed values
        y_x_test = dpoly_x(ximg, yimg)
        y_y_test = dpoly_y(ximg, yimg)

        assert np.allclose(y_x, y_x_test)
        assert np.allclose(y_y, y_y_test)
