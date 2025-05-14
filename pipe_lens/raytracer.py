import numpy as np

from numpy import ndarray, pi
from numpy.linalg import norm

from pipe_lens.raytracing_utils import uhp, roots_bhaskara, snell
from pipe_lens.acoustic_lens import AcousticLens
from pipe_lens.geometric_utils import Pipeline
from pipe_lens.transducer import Transducer

__all__ = ["RayTracer"]

FLOAT = np.float32


class RayTracer:
    def __init__(self, acoustic_lens: AcousticLens, pipeline: Pipeline, transducer: Transducer):
        self.transducer = transducer
        self.pipeline = pipeline
        self.acoustic_lens = acoustic_lens

    def solve(self, xf, zf, maxiter: int = 6):
        if isinstance(xf, (int, float)) and isinstance(zf, (int, float)):
            xf, zf = np.array([xf]), np.array([zf])

        Nfocus = len(xf)
        Nel = self.transducer.num_elem

        solution = self.__newton_batch(xf, zf, maxiter)

        return solution

    def get_tofs(self, xf, zf, maxiter: int = 6):
        n_elem = self.transducer.num_elem
        n_focii = len(xf)

        coord_elements = np.array([self.transducer.xt, self.transducer.zt]).T
        coords_reflectors = np.array([xf, zf]).T
        coords_lens = np.zeros(shape=(n_elem, 2, n_focii))
        coords_outer = np.zeros(shape=(n_elem, 2, n_focii))

        solution = self.solve(xf, zf, maxiter)
        for combined_idx in range(n_focii * n_elem):
            i = combined_idx // n_elem
            j = combined_idx % n_elem

            coords_lens[j, :, i] = solution[j]['xlens'][i], solution[j]['zlens'][i]
            coords_outer[j, :, i] = solution[j]['xpipe'][i], solution[j]['zpipe'][i]

        coord_elements_mat = np.tile(coord_elements[:, :, np.newaxis], (1, 1, n_focii))
        coord_reflectors_mat = np.tile(coords_reflectors[:, :, np.newaxis], (1, 1, n_elem))

        # Compute distances between points where refractions is expected.
        d1 = norm(coords_lens - coord_elements_mat, axis=1)  # distance between elements and lens
        d2 = norm(coords_lens - coords_outer, axis=1)  # distance between lens and pipe outer surface
        d3 = norm(coords_outer - coord_reflectors_mat.T, axis=1)  # distance between pipe outer surface and focus

        c1 = self.acoustic_lens.c1  # lens material
        c2 = self.acoustic_lens.c2  # coupling medium
        c3 = self.pipeline.c  # pipe material

        return d1 / c1 + d2 / c2 + d3 / c3

    def __newton_batch(self, xf: ndarray, yf: ndarray, iter: int, verbose=False) -> list:
        '''Calls the function newton() one time for each transducer element.
      The set of angles found for a given element are used as initial guess for
      the next one. Starts from the center of the transducer.'''

        N_elem = self.transducer.num_elem
        xc, yc = self.transducer.get_coords()

        results = [None] * N_elem
        results[N_elem // 2] = self.__newton(xc[N_elem // 2], yc[N_elem // 2], xf, yf)
        results[N_elem // 2 - 1] = self.__newton(xc[N_elem // 2 - 1], yc[N_elem // 2 - 1], xf, yf)
        for i in range(1, N_elem // 2):
            i_m = N_elem // 2 - i - 1
            i_p = N_elem // 2 + i

            alpha_init = np.arctan(results[i_p - 1]['xlens'] / results[i_p - 1]['zlens'])

            # Compute the optimal path for a given transducer (xc,yc) focus (xf, yf) pair:
            results[i_p] = self.__newton(xc[i_p], yc[i_p], xf, yf, alpha_init=alpha_init, iter=iter)
            bad_indices = results[i_p]['dist'] > 1e-8

            if verbose:
                if np.count_nonzero(bad_indices) > 0:
                    print('Bad indices found at ' + str(i_p) + ': ' + str(np.count_nonzero(bad_indices)))

            alpha_init = np.arctan(results[i_m + 1]['xlens'] / results[i_m + 1]['zlens'])
            results[i_m] = self.__newton(xc[i_m], yc[i_m], xf, yf, alpha_init=alpha_init, iter=iter)
            bad_indices = results[i_m]['dist'] > 1e-8

            if verbose:
                if np.count_nonzero(bad_indices) > 0:
                    print('Bad indices found at ' + str(i_m) + ': ' + str(np.count_nonzero(bad_indices)))
        return results

    def __newton(self, xc: float, yc: float, xf: ndarray, yf: ndarray, alpha_init=None, iter: int = 10) -> dict:
        '''Uses the Newton-Raphson method to compute the direction in which
      the transducer at (xc, yc) should fire in order to hit the "pixel"
      at (xf, yf). A dictionary is returned with the following information:

      (xcurve, ycurve): position where the ray hits the lens
      (xcirc, ycirc): position where the ray hits the circle (cylinder)
      (xin, yin): point on the ray closest to (xf, yf)
      dist: squared distance (xin, yin) to (xf, yf) (should be close to zero)
      maxdist: maximum squared distance (assuming an array of pixels was passed)
      mindist: minimum squared distance (assuming an array of pixels was passed)'''

        if alpha_init is None:
            alphaa = np.arctan(xf / yf)
        else:
            alphaa = alpha_init
        maxdist = list()
        mindist = list()
        for i in range(iter):
            dic, d1, d2 = self.__dist_and_derivatives(xc, yc, xf, yf, alphaa, eps=1e-4)
            alphaa -= d1 / d2
            alphaa[alphaa > self.acoustic_lens.alpha_max] = self.acoustic_lens.alpha_max * .9
            alphaa[alphaa < -self.acoustic_lens.alpha_max] = -self.acoustic_lens.alpha_max * .9
            maxdist.append(dic['dist'].max())
            mindist.append(dic['dist'].min())
        dic['maxdist'] = maxdist
        dic['mindist'] = mindist
        return dic

    def __dist_and_derivatives(self, xc, yc, xf, yf, acurve, eps=1e-5):
        '''Computes the squared distance using distalpha as well as the first and
      second derivatives of the squared distance with relation to alpha.'''
        dm = self.__distalpha(xc, yc, xf, yf, acurve - eps)['dist']
        dic = self.__distalpha(xc, yc, xf, yf, acurve)
        d0 = dic['dist']
        dp = self.__distalpha(xc, yc, xf, yf, acurve + eps)['dist']
        der1 = (dp - dm) * .5 / eps
        der2 = (dm - 2 * d0 + dp) / eps ** 2
        return dic, der1, der2

    def __distalpha(self, xc: float, zc: float, xf: ndarray, yf: ndarray, acurve: float):
        """For a shot fired from (xc, yc) in the direction x_y_from_alpha(alpha),
      this function computes the two refractions (from c1 to c2 and from
      c2 to c3) and then computes the squared distance between the ray at c3 and
      the "pixel" (xf, yf). A dictionary is returned with the following information:

      (xlens, ylens): position where the ray hits the lens
      (xcirc, ycirc): position where the ray hits the circle (pipeline)
      (xin, yin): point on the ray closest to (xf, yf)
      dist: squared distance (xin, yin) to (xf, yf)"""

        c1 = self.acoustic_lens.c1  # Wedge material
        c2 = self.acoustic_lens.c2  # Coupling medium
        c3 = self.pipeline.c  # Pipeline material

        # First ray from emitter to lens:
        xlens, ylens = self.acoustic_lens.xy_from_alpha(acurve)
        gamma1 = np.arctan((ylens - zc) / (xlens - xc))
        gamma1 = gamma1 + (gamma1 < 0) * pi  # incident angle
        gamma2, inc12, ref12 = snell(c1, c2, gamma1, self.acoustic_lens.dydx_from_alpha(acurve))  # refracted angle
        # Line equation which defines the ray within coupling medium (z = ax + b).
        a_line = np.tan(uhp(gamma2))
        b_line = ylens - a_line * xlens

        # Second ray a, b and c parameters:
        a = a_line ** 2 + 1
        b = 2 * a_line * b_line
        c = b_line ** 2 - self.pipeline.outer_radius ** 2
        xcirc1, xcirc2 = roots_bhaskara(a, b, c)
        ycirc1, ycirc2 = a_line * xcirc1 + b_line, a_line * xcirc2 + b_line
        upper = ycirc1 > ycirc2
        xcirc = xcirc1 * upper + xcirc2 * (1 - upper)
        ycirc = ycirc1 * upper + ycirc2 * (1 - upper)
        gamma3, inc23, ref23 = snell(c2, c3, gamma2, self.pipeline.dydx(xcirc))
        a3 = np.tan(gamma3)
        b3 = ycirc - a3 * xcirc
        xbottom = -b3 / a3
        a4 = -1 / a3
        b4 = yf - a4 * xf
        xin = (b4 - b3) / (a3 - a4)
        yin = a3 * xin + b3
        dist = (xin - xf) ** 2 + (yin - yf) ** 2
        dic = {'xlens': xlens, 'zlens': ylens, 'xpipe': xcirc, 'zpipe': ycirc, 'dist': dist,
               'interface_12': [inc12, ref12], "interface_23": [inc23, ref23]}
        return dic

    # TODO:
    # [ ] Add incidence and refraction angle to the dict. The goal is to compute transmission coefficients.

