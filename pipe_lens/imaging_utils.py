import numpy as np
import matplotlib.pyplot as plt
from bisect import bisect
from framework.post_proc import envelope


def convert_time2radius(tspan, t_outer, t_inner, cl_surf, cl_upper, cl_bottom=None, outer_r=70):
    # Converts the 'tspan' in us into a 'rspan' in mm, assuming that the outer-surface is located at 't_outer'
    # us, the inner-surface is located at 't_inner' us.
    if cl_bottom is None:
        cl_bottom = cl_upper
    if t_inner < t_outer:
        raise ValueError('t_inner must be smaller than t_outer')
    # Finds where the object profile begins and ends in index:
    tspan_copy = np.copy(tspan)
    beg_idx = np.argmin((tspan_copy - t_outer) ** 2)
    end_idx = np.argmin((tspan_copy - t_inner) ** 2)
    tspan_copy -= tspan_copy[beg_idx]
    tspan_copy *= -1

    # Converts the tspan into rspan
    rspan = np.zeros_like(tspan_copy)
    rspan[:beg_idx] = cl_bottom * (tspan_copy[:beg_idx] / 2) + outer_r
    rspan[beg_idx:end_idx] = cl_surf * (tspan_copy[beg_idx:end_idx] / 2) + outer_r
    rspan[end_idx:] = cl_upper * ((tspan_copy[end_idx:]) / 2)
    rspan[end_idx:] -= rspan[end_idx:].max()
    rspan[end_idx:] = rspan[end_idx:] + rspan[end_idx - 1]

    return rspan


def moving_average(vector, n=2):
    n_elem = vector.shape[0]
    output = np.zeros(shape=n_elem - n + 1)
    for i in range(0, n_elem - n + 1):
        output[i] = np.mean(vector[i:i + n])
    return output


def crop_ascan(ascan, t_span, t0=None, tf=None):
    if t0 is not None and tf is not None:
        t0_idx = np.argmin(np.power(t_span - t0, 2))
        tf_idx = np.argmin(np.power(t_span - tf, 2))
        return ascan[t0_idx:tf_idx, :]


def plot_echoes(t_base, t_echoes, n_echoes=3, color='blue', label='_', xbeg=-40, xend=40, alpha=.3):
    x = np.arange(xbeg, xend, 1e-1)
    for n in range(n_echoes):
        y = np.ones_like(x) * (t_base + t_echoes * (n + 1))
        plt.plot(x, y, ':', color=color, label=label, alpha=alpha)
    if label != "_":
        plt.legend()


def convert_time2idx(t, value):
    return np.argmin(np.power(t - value, 2))


def convert_corners_time2idx(t, ang, corner_in_time):
    north_east_corner_row, north_east_corner_column = corner_in_time[0]
    south_west_corner_row, south_west_corner_column = corner_in_time[1]
    corner_in_idx = \
        [
            (convert_time2idx(t, north_east_corner_row), convert_time2idx(ang, north_east_corner_column)),
            (convert_time2idx(t, south_west_corner_row), convert_time2idx(ang, south_west_corner_column))
        ]
    return corner_in_idx



def fwhm(sscan, r_span, theta_deg_span, corners_in, thresh=-6, drawSquare=True, time_grid=None):
    # Corner in [(r1, theta1), (r2, theta2)] in (mm, deg)

    # Converts (mm, deg) to (index, index):
    corners = convert_corners_time2idx(r_span, theta_deg_span, corners_in)  # Corner in index
    north_east_corner = corners[0]
    south_west_corner = corners[1]

    # Crops only the API Region of Interest (ROI):
    img_cropped = sscan[north_east_corner[0]:south_west_corner[0], north_east_corner[1]:south_west_corner[1]]

    # Obtain the maximum amplitude within the ROI:
    maximumAmplitude = np.max(img_cropped)

    # Creates a binary image where only pixels above thesh_db are highlighted
    pixels_above_threshold_within_roi = img_cropped >= maximumAmplitude * 10 ** (thresh / 20)
    pixels_above_threshold = np.zeros_like(sscan, dtype=bool)
    pixels_above_threshold[north_east_corner[0]:south_west_corner[0],
    north_east_corner[1]:south_west_corner[1]] += pixels_above_threshold_within_roi

    if np.any(pixels_above_threshold == True):
        maxLocationWithinROI = np.where(img_cropped == maximumAmplitude)
        maxAbsoluteLocation = maxLocationWithinROI[1] + north_east_corner[1]
    else:
        maxAbsoluteLocation = None

    # Compute the area of each pixel considering:
    # * All pixels above threshold belongs to the tube wall;
    # * The tube center is aligned with (0, 0) of the coordinate system;
    # * The sampling is constant along radius and angle axis;
    r_borders = moving_average(r_span)
    diff_i = r_borders[0] - r_span[0]
    diff_f = r_span[-1] - r_borders[-1]
    r_borders = np.append(r_span[0] - diff_i, r_borders)
    r_borders = np.append(r_borders, r_span[-1] + diff_f)

    #
    theta_borders = moving_average(theta_deg_span)
    diff_i = theta_borders[0] - theta_deg_span[0]
    diff_f = theta_deg_span[-1] - theta_borders[-1]
    theta_borders = np.append(theta_deg_span[0] - diff_i, theta_borders)
    theta_borders = np.append(theta_borders, theta_deg_span[-1] + diff_f)

    pixels_area = np.zeros_like(sscan)
    pixels_coords = np.zeros(shape=(*sscan.shape, 2))
    pixels_coords_2 = np.zeros_like(pixels_coords)
    for i in range(sscan.shape[0]):
        for j in range(sscan.shape[1]):
            inner_radius = r_borders[i]
            outer_radius = r_borders[i + 1]
            theta1 = theta_borders[j]
            theta2 = theta_borders[j + 1]
            diff_theta = np.abs(theta2 - theta1)

            #
            pixels_area[i, j] = (outer_radius * 2 - inner_radius * 2) * np.pi * diff_theta / 360

            pixel_radius = (outer_radius + inner_radius) / 2
            pixel_ang = (theta1 + theta2) / 2

            pixels_coords[i, j, 0] = pixel_ang

            if time_grid is None:
                pixels_coords[i, j, 1] = pixel_radius
            else:
                pixels_coords[i, j, 1] = float(time_grid[i])
    #



    estimated_width = pixels_coords[pixels_above_threshold, 0].max() - pixels_coords[pixels_above_threshold, 0].min()
    estimated_height = pixels_coords[pixels_above_threshold, 1].max() - pixels_coords[pixels_above_threshold, 1].min()

    #
    # import matplotlib
    # matplotlib.use('TkAgg')
    #
    #
    # plt.figure()
    # plt.pcolormesh(theta_deg_span, time_grid, pixels_above_threshold)
    # plt.show()

    cnr_ang_beg, cnr_ang_end = np.argmin(np.power(theta_deg_span + 10, 2)), np.argmin(np.power(theta_deg_span + 4, 2))
    cnr_radius_beg, cnr_radius_end = np.argmin(np.power(r_span - corners_in[0][0], 2)), np.argmin(np.power(r_span - corners_in[1][0], 2))

    # plt.imshow(sscan)
    Ib = sscan[cnr_radius_beg:cnr_radius_end, cnr_ang_beg:cnr_ang_end]
    Is = sscan[pixels_above_threshold]

    Is_mean = np.mean(Is)
    Ib_mean = np.mean(Ib)
    sigma_b = np.std(Ib)
    cnr = np.abs(Is_mean - Ib_mean) / sigma_b

    if drawSquare:
        width = 2
        scale_factor = int(sscan.shape[0] / sscan.shape[1])
        pixels_above_threshold[north_east_corner[0] - width * scale_factor: south_west_corner[0] + width * scale_factor,
        north_east_corner[1] - width: north_east_corner[1] + width] = 1

        pixels_above_threshold[north_east_corner[0] - width * scale_factor: south_west_corner[0] + width * scale_factor,
        south_west_corner[1] - width: south_west_corner[1] + width] = 1

        pixels_above_threshold[north_east_corner[0] - width * scale_factor: north_east_corner[0] + width * scale_factor,
        north_east_corner[1] - width: south_west_corner[1] + width] = 1

        pixels_above_threshold[south_west_corner[0] - width * scale_factor: south_west_corner[0] + width * scale_factor,
        north_east_corner[1] - width: south_west_corner[1] + width] = 1

    return estimated_width, estimated_height, maximumAmplitude, pixels_above_threshold, cnr

def image_in_db(img, apply_envelope=True):
    if apply_envelope:
        return 20 * np.log10(envelope(img, axis=0))
    else:
        return 20 * np.log10(img)


def compute_sscan(fmc, shifts_e, shifts_r):
    # From a given FMC apply the delays and compute the Summed-Scan (S-Scan):
    num_samples, num_elements, _, num_reflectors = fmc.shape
    num_laws = shifts_e.shape[0]

    if shifts_e.shape != shifts_r.shape:
        raise ValueError("Invalid shape for the delays")

    sscan = np.zeros(shape=(num_samples, num_laws, num_reflectors), dtype=float)
    signal_recepted_by_focus = np.zeros(shape=(num_samples, num_laws, num_reflectors), dtype=float)
    for scan_idx in range(num_laws):
        # Delay And Sum in emission:
        shift_e = shifts_e[scan_idx, :]
        rolled_fmc = np.zeros_like(fmc)
        for i in range(num_elements):
            rolled_fmc[:, i, :, :] = np.roll(fmc[:, i, :, :], int(shift_e[i]), axis=0)
        das_emission = np.sum(rolled_fmc, axis=1)
        signal_recepted_by_focus[:, scan_idx, :] = np.sum(das_emission, axis=1)

        # Delay And Sum in reception:
        shift_r = shifts_r[scan_idx, :]
        das = np.zeros_like(das_emission)
        for i in range(num_elements):
            das[:, i, :] = np.roll(das_emission[:, i, :], int(shift_r[i]), axis=0)
        ascan = np.sum(das, axis=1)
        sscan[:, scan_idx, :] = ascan
    return sscan, signal_recepted_by_focus
