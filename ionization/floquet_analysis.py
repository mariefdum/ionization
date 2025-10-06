import numpy as np
import qutip as qt
from joblib import parallel_backend, Parallel, delayed
from scipy.integrate import simps
import matplotlib.pyplot as plt


options = {
    "nsteps": 1e9,
    "atol": 1e-8,
    "rtol": 1e-06,
    "method": "bdf",
    "order": 5,
}


def format_parralel_outup(results, variable_vec, index):
    results_index = []
    for i in range(len(variable_vec)):
        results_index.append(results[i][index])
    return np.array(results_index).squeeze()


def generate_FA(Floquet_params, verbose=5):
    num_cores = Floquet_params["num_cores"]
    eps_d_vec = Floquet_params["eps_d_vec"]
    results = Parallel(n_jobs=num_cores, verbose=verbose)(
        delayed(floquet_eigs)(eps_d, Floquet_params) for eps_d in eps_d_vec
    )

    index_tot = format_parralel_outup(results, eps_d_vec, 0)
    f_energies_tot = format_parralel_outup(results, eps_d_vec, 1)
    f_modes_0_tot = format_parralel_outup(results, eps_d_vec, 2)

    return index_tot, f_energies_tot, f_modes_0_tot


def H_drive_coeff(t, args=None):
    omega_d = args["omega_d"]
    phi = args["phi"]
    return np.cos(omega_d * t + phi)


def floquet_eigs(eps_d, Floquet_params):
    n_t = Floquet_params["n_t"]
    I_t = Floquet_params["I_t"]
    H_t = Floquet_params["H_t"]
    n_g = Floquet_params["n_g"]
    omega_d = Floquet_params["omega_d"]
    T = 2 * np.pi / omega_d

    H_drive = eps_d * (n_t - I_t * n_g)
    H = [H_t, [H_drive, H_drive_coeff]]

    Floquet_obj = qt.FloquetBasis(H, T, args=Floquet_params)  # ,options=options)
    f_modes_0 = evecs_qutip_to_numpy(Floquet_obj.mode(0))
    f_energies = Floquet_obj.e_quasi
    index = f_energies.argsort()

    return [index, f_energies, f_modes_0]


# STATE TRACKING
def track(q_max, f_energies, f_modes_0, Floquet_params, Transmon_params, eigs_t):
    eps_d_vec = Floquet_params["eps_d_vec"]
    omega_d = Floquet_params["omega_d"]
    n_trunc = Transmon_params["n_trunc"]

    f_modes_0_sorted = []
    f_energies_sorted = []
    overlap_max = []
    index_sorted = []

    f_modes_0 = np.array(f_modes_0).squeeze()

    for q in range(q_max):
        f_modes_q = []
        f_energies_q = []
        overlap_max_q = []
        index_q = []

        f_modes_q.append(qt.basis(n_trunc, q).full().squeeze())
        f_energies_q.append((eigs_t[q] + omega_d / 2) % omega_d - omega_d / 2)
        index_q.append(q)

        for j in range(len(eps_d_vec) - 1):
            overlap_i = 0
            index = 0
            for i in range(n_trunc):
                overlap = (
                    np.abs(qt.Qobj(f_modes_q[j]).overlap(qt.Qobj(f_modes_0[j + 1, i])))
                    ** 2
                )
                if overlap > overlap_i:
                    index = i
                    overlap_i = overlap
                    f_mode_next = f_modes_0[j + 1, i]
            overlap_max_q.append(overlap_i)
            f_modes_q.append(f_mode_next)
            f_energies_q.append(f_energies[j + 1, index])
            index_q.append(index)

        overlap_max.append(overlap_max_q)
        f_modes_0_sorted.append(f_modes_q)
        f_energies_sorted.append(f_energies_q)
        index_sorted.append(index_q)

    return (
        overlap_max,
        f_modes_0_sorted,
        f_energies_sorted,
        index_sorted,
    )  


def generate_FA_time_av(index_sorted, Floquet_params, Transmon_params):
    eps_d_vec = Floquet_params["eps_d_vec"]
    num_cores = Floquet_params["num_cores"]
    n_trunc = Transmon_params["n_trunc"]
    i_vec = np.arange(0, len(eps_d_vec))
    init_cons = np.vstack((i_vec, eps_d_vec)).T
    results = Parallel(n_jobs=num_cores, verbose=3)(
        delayed(time_av_H)(init, Floquet_params, Transmon_params) for init in init_cons
    )

    av_E_transmon = []
    for i in range(len(eps_d_vec)):
        av_E_transmon.append(results[i][0])
    av_E_transmon = np.array(av_E_transmon)

    av_E = []
    for i in range(len(eps_d_vec)):
        av_E.append(results[i][1])
    av_E = np.array(av_E)

    av_N_t = []
    for i in range(len(eps_d_vec)):
        av_N_t.append(results[i][2])
    av_N_t = np.array(av_N_t)

    av_E_transmon_sorted = []
    av_E_sorted = []
    av_N_t_sorted = []

    for q in range(n_trunc):
        av_E_transmon_q = []
        av_E_q = []
        av_N_t_q = []
        for i in range(len(eps_d_vec)):
            av_E_transmon_q.append(av_E_transmon[i, index_sorted[q][i]])
            av_E_q.append(av_E[i, index_sorted[q][i]])
            av_N_t_q.append(av_N_t[i, index_sorted[q][i]])
        av_E_transmon_sorted.append(np.array(av_E_transmon_q))
        av_E_sorted.append(np.array(av_E_q))
        av_N_t_sorted.append(np.array(av_N_t_q))

    return av_E_transmon_sorted, av_E_sorted, av_N_t_sorted


# TIME AVERAGE
def time_av_H(init, Floquet_params, Transmon_params):
    n_trunc = Transmon_params["n_trunc"]
    N_times = Floquet_params["N_times"]
    N_periods = Floquet_params["N_periods"]
    T = 2 * np.pi / Floquet_params["omega_d"]
    n_t = Floquet_params["n_t"]
    I_t = Floquet_params["I_t"]
    H_t = Floquet_params["H_t"]
    n_g = Floquet_params["n_g"]
    times = np.linspace(0, T * N_periods, N_times, endpoint=False)
    i = int(init[0])
    eps_d = init[1]

    H_drive = eps_d * (n_t - I_t * n_g)
    H = [H_t, [H_drive, H_drive_coeff]]

    f_modes_0, f_energies = qt.floquet_modes(H, T, Floquet_params, options=options)
    # qt.fileio.qsave(f_modes_0,path+'f_modes_0/f_modes_0_'+str(i))

    f_modes_t = qt.floquet_modes_table(
        f_modes_0, f_energies, times, H, T, Floquet_params, options=options
    )

    expect_energy_transmon = np.zeros((n_trunc, N_times))
    expect_energy = np.zeros((n_trunc, N_times))
    N_t_all = np.zeros((n_trunc, N_times))

    for n, t in enumerate(times):
        modes_t = qt.floquet_modes_t_lookup(f_modes_t, t, T)
        expect_energy_transmon[:, n] = [
            np.real(expect_t_transmon(mode, t, Floquet_params)) for mode in modes_t
        ]
        expect_energy[:, n] = [
            np.real(expect_t(mode, t, Floquet_params, eps_d)) for mode in modes_t
        ]
        N_t_all[:, n] = [np.real(N_t(mode, Transmon_params)) for mode in modes_t]

    av_E_transmon = np.zeros((n_trunc,))
    av_E = np.zeros((n_trunc,))
    N_t_av = np.zeros((n_trunc,))

    for n in range(n_trunc):
        av_E_transmon[n] = simps(expect_energy_transmon[n, :], times) / T / N_periods
        av_E[n] = simps(expect_energy[n, :], times) / T / N_periods
        N_t_av[n] = simps(N_t_all[n, :], times) / T / N_periods

    return [av_E_transmon, av_E, N_t_av, expect_energy_transmon, expect_energy]


def expect_t(mode, t, Floquet_params, eps_d):
    n_t = Floquet_params["n_t"]
    I_t = Floquet_params["I_t"]
    H_t = Floquet_params["H_t"]
    n_g = Floquet_params["n_g"]
    omega_d = Floquet_params["omega_d"]
    return qt.expect(H_t + eps_d * (n_t - I_t * n_g) * np.cos(omega_d * t), mode)


def expect_t_transmon(mode, t, Floquet_params):
    H_t = Floquet_params["H_t"]
    return qt.expect(H_t, mode)


def N_t(mode, Transmon_params):
    n_trunc = Transmon_params["n_trunc"]
    return (np.squeeze(np.abs(mode[:]) ** 2)).dot(np.arange(0, n_trunc))


def N_t_q(q, sweep_param_vec, f_modes_0_sorted_tot, Transmon_params):
    N_t_q = []
    for sweep_param_index, sweep_param in enumerate(sweep_param_vec):
        N_t_q.append(N_t(f_modes_0_sorted_tot[sweep_param_index][q], Transmon_params))
    N_t_q = np.array(N_t_q)
    return N_t_q


def N_t_min(
    q,
    sweep_param_vec,
    q_list_tot,
    index_match_list_tot,
    f_modes_0_sorted_tot,
    Transmon_params,
):
    N_t_min = []
    for sweep_param_index, sweep_param in enumerate(sweep_param_vec):
        N_t_min_index = []
        q_list = q_list_tot[sweep_param_index][q]
        index_match_list = index_match_list_tot[sweep_param_index][q]
        for index_q, new_q in enumerate(q_list):
            index_i = index_match_list[index_q]
            index_f = index_match_list[index_q + 1]
            N_t_min_index.append(
                N_t(f_modes_0_sorted_tot[sweep_param_index][new_q], Transmon_params)[
                    index_i:index_f
                ]
            )
        N_t_min.append(np.hstack(N_t_min_index))
    return N_t_min


def N_t_min_abs(sweep_param_vec, f_modes_0_sorted_tot, Transmon_params):
    N_t_tot = []
    n_trunc = Transmon_params["n_trunc"]
    for q in range(n_trunc):
        N_t = N_t_q(q, sweep_param_vec, f_modes_0_sorted_tot, Transmon_params)
        N_t_tot.append(N_t)
    N_t_tot = np.array(N_t_tot)
    N_t_min = np.min(N_t_tot, axis=0)
    return N_t_min

def N_t_q_nosweep(q, f_modes_0_sorted_tot, Transmon_params):
    N_t_q = N_t(f_modes_0_sorted_tot[q], Transmon_params)
    N_t_q = np.array(N_t_q)
    return N_t_q


def N_t_min_nosweep(
    q, q_list_tot, index_match_list_tot, f_modes_0_sorted_tot, Transmon_params
):
    N_t_min_index = []
    q_list = q_list_tot[0]
    index_match_list = index_match_list_tot[0]
    for index_q, q in enumerate(q_list):
        index_i = index_match_list[index_q]
        index_f = index_match_list[index_q + 1]
        N_t_min_index.append(
            N_t(f_modes_0_sorted_tot[q], Transmon_params)[index_i:index_f]
        )
    N_t_min = np.hstack(N_t_min_index)
    return N_t_min


def N_t_min_abs_nosweep(f_modes_0_sorted_tot, Transmon_params):
    N_t_tot = []
    n_trunc = Transmon_params["n_trunc"]
    for q in range(n_trunc):
        N_t = N_t_q_nosweep(q, f_modes_0_sorted_tot, Transmon_params)
        N_t_tot.append(N_t)
    N_t_tot = np.array(N_t_tot)
    N_t_min = np.min(N_t_tot, axis=0)
    return N_t_min, N_t_tot


def plot_quasi_energies(
    f_energies_ordered,
    eps_d_vec,
    n_trunc,
    omega_r,
    palette,
    text="yes",
    xscale="n",
):
    q_max_plot = n_trunc

    for q in range(q_max):
        f_energies_ordered_q = np.array(f_energies_ordered[q])
        x_i = 0
        for i in range(len(f_energies_ordered_q) - 1):
            if (
                np.abs((f_energies_ordered_q[i] - f_energies_ordered_q[i + 1]))
                > 0.9 * omega_r
            ):
                x_f = i
                x = eps_d_vec[x_i:x_f] / 2 / g
                y = np.array((f_energies_ordered[q])[x_i:x_f]) / 2 / np.pi / 10**9
                if xscale == "n":
                    plt.plot(
                        x**2,
                        y,
                        "-",
                        markersize=1,
                        linewidth=0.5,
                        label=q,
                        color=palette[q % len(palette)],
                    )
                elif xscale == "eps_d":
                    plt.plot(
                        x,
                        y,
                        "-",
                        markersize=1,
                        linewidth=0.5,
                        label=q,
                        color=palette[q % len(palette)],
                    )
                x_i = i + 1
        x = eps_d_vec[x_i:] / 2 / g
        y = np.array((f_energies_ordered[q])[x_i:]) / 2 / np.pi / 10**9
        if xscale == "n":
            plt.plot(
                x**2,
                y,
                "-",
                markersize=1,
                linewidth=0.5,
                label=q,
                color=palette[q % len(palette)],
            )
        elif xscale == "eps_d":
            plt.plot(
                x,
                y,
                "-",
                markersize=1,
                linewidth=0.5,
                label=q,
                color=palette[q % len(palette)],
            )

    if xscale == "n":
        plt.xlabel(r"$\overline{n}$")
    elif xscale == "eps_d":
        plt.xlabel(r"$\sqrt{\overline{n}}$")
    plt.ylabel("Floquet quasi-energies [GHz]")
    plt.title("Floquet Spectrum")


def evecs_qutip_to_numpy(evecs_qutip):
    evecs_numpy = np.array([x.full() for x in evecs_qutip])
    return evecs_numpy


def calculate_overlap_tot(n_trunc, f_modes_0_sorted, eps_d_vec):

    overlap_tot = []

    for eps_d_index in range(len(eps_d_vec)):
        if eps_d_index < len(eps_d_vec) - 1:
            f_modes_0_eps_i = np.array(f_modes_0_sorted)[:, eps_d_index, :]
            f_modes_0_eps_f = np.array(f_modes_0_sorted)[:, eps_d_index + 1, :]

            overlap_eps_d_index = (
                np.abs(f_modes_0_eps_i.dot(f_modes_0_eps_f.T.conj())) ** 2
            )
            overlap_tot.append(overlap_eps_d_index)

    overlap_tot = np.array(overlap_tot)
    return overlap_tot


####################################################### FIND RESONANCES WITH OVERLAPS, BY LOOKING AT OVERLAPS WITH OTHER BRANCHES #############################################################################


def find_local_maxima(arr, index_min, value_min):

    local_maxima = []
    n = len(arr)

    if n == 0:
        return local_maxima

    # Check if the first element is a local maximum
    # if n > 1 and arr[0] > arr[1]:
    #    local_maxima.append((0, arr[0]))

    # Check for local maxima in the middle of the list
    for i in range(1, n - 1):
        if arr[i] > arr[i - 1] and arr[i] > arr[i + 1]:
            if arr[i] > value_min:
                if i > index_min:
                    local_maxima.append((i, arr[i]))

    # Check if the last element is a local maximum
    # if n > 1 and arr[-1] > arr[-2]:
    #    local_maxima.append((n - 1, arr[-1]))

    return local_maxima


def find_local_minima(arr, index_min, value_max):

    local_minima = []
    n = len(arr)

    if n == 0:
        return local_minima

    # Check if the first element is a local maximum
    # if n > 1 and arr[0] > arr[1]:
    #    local_maxima.append((0, arr[0]))

    # Check for local maxima in the middle of the list
    for i in range(1, n - 1):
        if arr[i] < arr[i - 1] and arr[i] < arr[i + 1]:
            if arr[i] < value_max:
                if i > index_min:
                    local_minima.append((i, arr[i]))

    # Check if the last element is a local maximum
    # if n > 1 and arr[-1] > arr[-2]:
    #    local_maxima.append((n - 1, arr[-1]))

    return local_minima


def sort_by_nth_column(combined_array, n):
    """
    Sorts an array by the third column.

    Parameters:
    combined_array (numpy.ndarray): The array to sort.

    Returns:
    numpy.ndarray: The sorted array.
    """
    sorted_indices = combined_array[:, n].argsort()
    sorted_array = combined_array[sorted_indices]
    return sorted_array


def filter_pairs(arr):
    # Step 1: Group by the first value
    groups = defaultdict(list)
    for item in arr:
        groups[item[0]].append(item)

    # Step 2: Check and retain valid pairs
    result = []
    for key, items in groups.items():
        pair_count = defaultdict(int)

        # Count occurrences of each (second, third) pair
        for item in items:
            pair = (item[1], item[2])
            pair_count[pair] += 1

        # Retain pairs that occur exactly twice
        valid_pairs = {pair for pair, count in pair_count.items() if count == 2}

        # Filter out invalid pairs
        for item in items:
            pair = (item[1], item[2])
            if pair in valid_pairs:
                result.append(item)

    return result


def find_matching_indices(array_m2, array_n2, n_g):
    """
    Finds the indices where the first value of the m-th row in array_m2
    matches the first value of the n-th row in array_n2.

    Parameters:
    array_m2 (numpy.ndarray): An array of size (m x 2).
    array_n2 (numpy.ndarray): An array of size (n x 2).

    Returns:
    list of tuples: A list of tuples where each tuple contains the indices (i, j)
                    indicating that array_m2[i, 0] == array_n2[j, 0].
    """
    matching_indices = []

    for i in range(array_m2.shape[0]):
        matching_j_indices = []

        matched = False
        # Find all matching indices j
        for j in range(array_n2.shape[0]):
            if array_m2[i, 0] == array_n2[j, 0]:
                matching_j_indices.append(j)
                matched = True

        # If there are matching indices, find the one with the maximal second value
        if matching_j_indices:
            max_j = max(matching_j_indices, key=lambda j: array_n2[j, 1])
            matching_indices.append((i, max_j))

        if matched == False and n_g != 0:
            # Calculate the differences
            differences = np.abs(array_n2[:, 0] - array_m2[i, 0])
            # Get the indices of the two smallest differences
            closest_indices = np.sort(np.argsort(differences)[:2])
            matching_indices.append((i, closest_indices[0]))
            if len(closest_indices) > 1:
                matching_indices.append((i, closest_indices[1]))

    return matching_indices


def remove_rows(array):
    """
    Removes rows j from the array if there is another row i such that
    row_i[0] = row_j[0], row_i[1] = row_j[2], and row_i[2] = row_j[1].

    Parameters:
    array (numpy.ndarray): The input array of size (m x 3).

    Returns:
    numpy.ndarray: The resulting array after removing the specified rows.
    """
    to_remove = np.full(array.shape[0], False)

    for i in range(array.shape[0]):
        for j in range(array.shape[0]):
            if (
                i != j
                and array[i, 1] < array[j, 1]
                and array[i, 0] == array[j, 0]
                and array[i, 1] == array[j, 2]
                and array[i, 2] == array[j, 1]
            ):
                to_remove[j] = True
    return array[~to_remove]


def has_local_extrema(arr):
    n = len(arr)
    if n < 3:
        return (
            False  # Array is too short to have local extrema excluding the endpoints.
        )

    for i in range(1, n - 1):
        if arr[i] > arr[i - 1] and arr[i] > arr[i + 1]:
            return True  # Local maxima found
        elif arr[i] < arr[i - 1] and arr[i] < arr[i + 1]:
            return True  # Local minima found

    return False  # No local maxima or minima found excluding the endpoints.


def find_fwhm_half(y, peak_index):

    peak_value = y[peak_index]

    # Determine the half-maximum value
    half_max = peak_value / 2.0

    # Find the crossing points on both sides of the peak
    left_side = y[:peak_index]
    right_side = y[peak_index:]

    left_crossing = np.where(left_side <= half_max)[0]
    right_crossing = np.where(right_side <= half_max)[0]

    if len(left_crossing) == 0 or len(right_crossing) == 0:
        fwhm = 0  # No crossing found, FWHM cannot be determined for this peak
    else:
        # Closest points to the peak where the signal crosses the half-maximum value
        left_crossing_point = left_crossing[-1]
        right_crossing_point = right_crossing[0] + peak_index

        # Calculate the widths on both sides
        fwhm_left = peak_index - left_crossing_point
        fwhm_right = right_crossing_point - peak_index

        # Choose the smallest width
        fwhm = min(fwhm_left, fwhm_right)

    return fwhm


def find_resonance_overlaps(
    n_g,
    n_trunc,
    eps_d_vec,
    f_energies_sorted,
    N_t_sorted,
    overlap_max,
    overlap_tot,
    param_resonances,
):

    index_min = param_resonances["index_min"]
    value_min = param_resonances["value_min"]
    ratio_v = param_resonances["ratio_v"]
    FWHM_coef = param_resonances["FWHM_coef"]
    FWHM_size_min = param_resonances["FWHM_size_min"]
    FWHM_overlap_min = param_resonances["FWHM_overlap_min"]

    param_resonances["FWHM_overlap_min"]

    value_max = (1 - ratio_v * value_min,)

    overlap_tot_max = []
    for q_i in range(n_trunc):
        overlap_tot_max_q = []
        for q_f in range(n_trunc):
            overlap_tot_max_q.append(
                np.array(
                    find_local_maxima(
                        overlap_tot[:, q_i, q_f],
                        index_min=index_min,
                        value_min=value_min,
                    )
                )
            )
        overlap_tot_max.append(overlap_tot_max_q)

    overlap_tot_min = []
    for q_i in range(n_trunc):
        overlap_tot_min.append(
            np.array(
                find_local_minima(
                    overlap_tot[:, q_i, q_i], index_min=index_min, value_max=value_max
                )
            )
        )

    res_q = []
    for q_i in range(n_trunc):
        overlap_tot_max_q = np.empty((0, 3))
        for q_f in range(n_trunc):
            if q_i != q_f:
                if overlap_tot_max[q_i][q_f].size != 0:
                    overlap_tot_max_q = np.vstack(
                        [
                            overlap_tot_max_q,
                            np.hstack(
                                [
                                    overlap_tot_max[q_i][q_f],
                                    (
                                        q_f * np.ones(len(overlap_tot_max[q_i][q_f]))
                                    ).reshape(-1, 1),
                                ]
                            ),
                        ]
                    )

        overlap_tot_max_q_sort = sort_by_nth_column(overlap_tot_max_q, n=0)
        if overlap_tot_max_q_sort.shape[0] > 0:
            indices_q = find_matching_indices(
                overlap_tot_min[q_i], overlap_tot_max_q_sort, n_g
            )
            for i in range(len(indices_q)):
                res_q.append(
                    [
                        int(overlap_tot_max_q_sort[indices_q[i][1]][0]),
                        q_i,
                        int(overlap_tot_max_q_sort[indices_q[i][1]][2]),
                        overlap_tot_min[q_i][indices_q[i][0]][1],
                        overlap_tot_max_q_sort[indices_q[i][1]][1],
                    ]
                )
                # res_q.append([int(overlap_tot_min[q_i][indices_q[i][0]][0]),q_i,int(overlap_tot_max_q_sort[indices_q[i][1]][2]),overlap_tot_min[q_i][indices_q[i][0]][1],overlap_tot_max_q_sort[indices_q[i][1]][1]])

    # Determine unique rows based on all but the last two columns
    # print(res_q)
    if res_q == []:
        return [], [], [], [], f_energies_sorted, N_t_sorted, []

    else:
        _, unique_indices = np.unique(
            np.array(res_q)[:, :-2], axis=0, return_index=True
        )

        # Use the unique indices to get the unique rows
        res_q_unique = np.array(res_q)[sorted(unique_indices)]

        res_q_sort = sort_by_nth_column(np.array(res_q_unique), n=0)
        res_q_sort_clean = remove_rows(res_q_sort)

        swap_statements = []
        for i in range(len(res_q_sort_clean)):

            q0, q1, n = (
                int(res_q_sort_clean[i, 1]),
                int(res_q_sort_clean[i, 2]),
                int(res_q_sort_clean[i, 0]),
            )
            swap_statement = True

            # check if this is a unavoided crossing
            for delta in [-2, -1, 1, 2]:
                if (
                    f_energies_sorted[q0][n - delta] > f_energies_sorted[q1][n - delta]
                    and f_energies_sorted[q0][n + delta]
                    < f_energies_sorted[q1][n + delta]
                ):
                    swap_statement = False

            if swap_statement == True:
                N_t_sorted_q0 = N_t_sorted[q0][n]
                N_t_sorted_q1 = N_t_sorted[q1][n]
                mean = (N_t_sorted_q0 + N_t_sorted_q1) / 2

                signal = overlap_tot[:, q0, q1]
                FWHM = int(find_fwhm_half(signal, n)) * 3

                if has_local_extrema(N_t_sorted[q0][n - FWHM : n + FWHM]):
                    if has_local_extrema(N_t_sorted[q1][n - FWHM : n + FWHM]):
                        if N_t_sorted_q0 < N_t_sorted_q1:
                            if np.all(
                                N_t_sorted[q0][n - FWHM : n + FWHM] < mean
                            ) or np.all(N_t_sorted[q1][n - FWHM : n + FWHM] > mean):
                                swap_statement = False
                        if N_t_sorted_q0 > N_t_sorted_q1:
                            if np.all(
                                N_t_sorted[q0][n - FWHM : n + FWHM] > mean
                            ) or np.all(N_t_sorted[q0][n - FWHM : n + FWHM] < mean):
                                swap_statement = False

                index_min = np.max([n - FWHM, 0])
                index_max = np.min([n + FWHM, len(N_t_sorted[q0])])
                comp = (
                    N_t_sorted[q0][index_min:index_max]
                    - N_t_sorted[q1][index_min:index_max]
                )

                if N_t_sorted_q0 < N_t_sorted_q1:
                    if np.all(comp < 0):
                        swap_statement = False
                if N_t_sorted_q0 > N_t_sorted_q1:
                    if np.all(comp > 0):
                        swap_statement = False

                if FWHM < FWHM_size_min and overlap_tot[n, q0, q1] < FWHM_overlap_min:
                    swap_statement = False

            # print(q0,q1,n,swap_statement)

            swap_statements.append(swap_statement)

        index_crit_tot = []
        branch_tot = []
        gap_tot = []

        for q in range(n_trunc):

            current_q = q
            index_crit_q = []
            branch_q = []
            gap_q = []

            branch_q.append(current_q)
            for i in range(len(res_q_sort_clean)):
                [n, q0, q1] = (res_q_sort_clean[i][0:3]).astype(int)
                gap = np.abs(f_energies_sorted[q0][n] - f_energies_sorted[q1][n])
                if q0 == current_q:
                    index_crit_q.append(n)
                    if swap_statements[i] == True:
                        current_q = q1
                        gap_q.append(gap)
                    else:
                        gap_q.append(0)
                    branch_q.append(current_q)
                elif q1 == current_q:
                    index_crit_q.append(n)
                    if swap_statements[i] == True:
                        current_q = q0
                        gap_q.append(gap)
                    else:
                        gap_q.append(0)
                    branch_q.append(current_q)

            index_crit_tot.append(index_crit_q)
            branch_tot.append(branch_q)
            gap_tot.append(gap_q)

        N_t_sorted_q_tot = []
        f_energies_sorted_q_tot = []

        for q in range(n_trunc):
            N_t_sorted_q = []
            f_energies_sorted_q = []
            for i in range(len(branch_tot[q])):
                if len(branch_tot[q]) > 1:
                    if i == 0:
                        ind_i = 0
                        ind_f = index_crit_tot[q][i]
                    elif i == len(branch_tot[q]) - 1:
                        ind_i = index_crit_tot[q][i - 1]
                        ind_f = len(overlap_max[q]) + 1
                    else:
                        ind_i = index_crit_tot[q][i - 1]
                        ind_f = index_crit_tot[q][i]
                else:
                    ind_i = 0
                    ind_f = len(overlap_max[q]) + 1

                N_t_sorted_q.append(N_t_sorted[branch_tot[q][i]][ind_i:ind_f])
                f_energies_sorted_q.append(
                    f_energies_sorted[branch_tot[q][i]][ind_i:ind_f]
                )

            # Concatenate the sorted sub-arrays into a single list
            N_t_sorted_q = [item for list in N_t_sorted_q for item in list]
            N_t_sorted_q_tot.append(N_t_sorted_q)

            f_energies_sorted_q = [
                item for list in f_energies_sorted_q for item in list
            ]
            f_energies_sorted_q_tot.append(f_energies_sorted_q)

        return (
            res_q_sort_clean,
            swap_statements,
            index_crit_tot,
            branch_tot,
            f_energies_sorted_q_tot,
            N_t_sorted_q_tot,
            gap_tot,
        )


####################################################### FIND RESONANCES WITH OVERLAPS, BY LOOKING AT OVERLAPS WITH OTHER BRANCHES (old version)
#############################################################################

# def evecs_qutip_to_numpy(evecs_qutip):
#     evecs_numpy = np.block([x.full() for x in evecs_qutip])
#     return evecs_numpy

# def find_local_maxima(arr,index_min,value_min):

#     local_maxima = []
#     n = len(arr)

#     if n == 0:
#         return local_maxima

#     # Check if the first element is a local maximum
#     #if n > 1 and arr[0] > arr[1]:
#     #    local_maxima.append((0, arr[0]))

#     # Check for local maxima in the middle of the list
#     for i in range(1, n - 1):
#         if arr[i] > arr[i - 1] and arr[i] > arr[i + 1]:
#             if arr[i]>value_min:
#                 if i>index_min:
#                     local_maxima.append((i, arr[i]))

#     # Check if the last element is a local maximum
#     #if n > 1 and arr[-1] > arr[-2]:
#     #    local_maxima.append((n - 1, arr[-1]))

#     return local_maxima


# def find_local_minima(arr,index_min,value_max):

#     local_minima = []
#     n = len(arr)

#     if n == 0:
#         return local_minima

#     # Check if the first element is a local maximum
#     #if n > 1 and arr[0] > arr[1]:
#     #    local_maxima.append((0, arr[0]))

#     # Check for local maxima in the middle of the list
#     for i in range(1, n - 1):
#         if arr[i] < arr[i - 1] and arr[i] < arr[i + 1]:
#             if arr[i]<value_max:
#                 if i>index_min:
#                     local_minima.append((i, arr[i]))

#     # Check if the last element is a local maximum
#     #if n > 1 and arr[-1] > arr[-2]:
#     #    local_maxima.append((n - 1, arr[-1]))

#     return local_minima


# def sort_by_nth_column(combined_array,n):
#     """
#     Sorts an array by the third column.

#     Parameters:
#     combined_array (numpy.ndarray): The array to sort.

#     Returns:
#     numpy.ndarray: The sorted array.
#     """
#     sorted_indices = combined_array[:, n].argsort()
#     sorted_array = combined_array[sorted_indices]
#     return sorted_array


# def filter_pairs(arr):
#     # Step 1: Group by the first value
#     groups = defaultdict(list)
#     for item in arr:
#         groups[item[0]].append(item)

#     # Step 2: Check and retain valid pairs
#     result = []
#     for key, items in groups.items():
#         pair_count = defaultdict(int)

#         # Count occurrences of each (second, third) pair
#         for item in items:
#             pair = (item[1], item[2])
#             pair_count[pair] += 1

#         # Retain pairs that occur exactly twice
#         valid_pairs = {pair for pair, count in pair_count.items() if count == 2}

#         # Filter out invalid pairs
#         for item in items:
#             pair = (item[1], item[2])
#             if pair in valid_pairs:
#                 result.append(item)

#     return result

# def find_matching_indices(array_m2, array_n2,n_g):
#     """
#     Finds the indices where the first value of the m-th row in array_m2
#     matches the first value of the n-th row in array_n2.

#     Parameters:
#     array_m2 (numpy.ndarray): An array of size (m x 2).
#     array_n2 (numpy.ndarray): An array of size (n x 2).

#     Returns:
#     list of tuples: A list of tuples where each tuple contains the indices (i, j)
#                     indicating that array_m2[i, 0] == array_n2[j, 0].
#     """
#     matching_indices = []

#     for i in range(array_m2.shape[0]):
#         matching_j_indices = []

#         matched=False
#         # Find all matching indices j
#         for j in range(array_n2.shape[0]):
#             if array_m2[i, 0] == array_n2[j, 0]:
#                 matching_j_indices.append(j)
#                 matched=True

#         # If there are matching indices, find the one with the maximal second value
#         if matching_j_indices:
#             max_j = max(matching_j_indices, key=lambda j: array_n2[j, 1])
#             matching_indices.append((i, max_j))

#         if matched==False and n_g!=0:
#             # Calculate the differences
#             differences = np.abs(array_n2[:, 0] - array_m2[i, 0])
#             # Get the indices of the two smallest differences
#             closest_indices = np.sort(np.argsort(differences)[:2])
#             matching_indices.append((i,closest_indices[0]))
#             if len(closest_indices)>1:
#                 matching_indices.append((i,closest_indices[1]))

#     return matching_indices

# def filter_rows(array):
#     result = []
#     for i, row in enumerate(array):
#         a, b, c, _, _ = row
#         # Check if there exists another row with values [a, c, b, _, _]
#         found_matching_row = False
#         for j, other_row in enumerate(array):
#             if i != j and other_row[0] == a and other_row[1] == c and other_row[2] == b:
#                 found_matching_row = True
#                 break
#         if found_matching_row:
#             result.append(row)
#     return np.array(result)


# def remove_rows(array):
#     """
#     Removes rows j from the array if there is another row i such that
#     row_i[0] = row_j[0], row_i[1] = row_j[2], and row_i[2] = row_j[1].

#     Parameters:
#     array (numpy.ndarray): The input array of size (m x 3).

#     Returns:
#     numpy.ndarray: The resulting array after removing the specified rows.
#     """
#     to_remove = np.full(array.shape[0], False)

#     for i in range(array.shape[0]):
#         for j in range(array.shape[0]):
#             if i != j and array[i, 1]<array[j, 1] and array[i, 0] == array[j, 0] and array[i, 1] == array[j, 2] and array[i, 2] == array[j, 1]:
#                 to_remove[j] = True
#     return array[~to_remove]

# def has_local_extrema(arr):
#     n = len(arr)
#     if n < 3:
#         return False  # Array is too short to have local extrema excluding the endpoints.

#     for i in range(1, n-1):
#         if arr[i] > arr[i-1] and arr[i] > arr[i+1]:
#             return True  # Local maxima found
#         elif arr[i] < arr[i-1] and arr[i] < arr[i+1]:
#             return True  # Local minima found

#     return False  # No local maxima or minima found excluding the endpoints.


# def find_resonance_overlaps(n_g,n_trunc,eps_d_vec,f_energies_sorted,f_modes_0_sorted,N_t_sorted,overlap_max,index_min=3*0,ratio_v=10,value_min=1e-6,delta_patch=10):
#     value_max=1-ratio_v*value_min,

#     f_modes_0_sorted_np=[]
#     for q in range(n_trunc):
#         f_modes_0_sorted_np.append(evecs_qutip_to_numpy(f_modes_0_sorted[q]))
#     f_modes_0_sorted_np=np.array(f_modes_0_sorted_np)

#     overlap_tot=[]
#     for eps_d_index,eps_d in enumerate(eps_d_vec):
#         if eps_d_index<len(eps_d_vec)-10:
#             f_modes_0_eps_i=(f_modes_0_sorted_np[:,:,eps_d_index])
#             f_modes_0_eps_f=(f_modes_0_sorted_np[:,:,eps_d_index+1])
#             overlap_eps_d_index=np.abs(f_modes_0_eps_i.dot(f_modes_0_eps_f.T.conj()))**2
#             overlap_tot.append(overlap_eps_d_index)
#     overlap_tot=np.array(overlap_tot)

#     overlap_tot_max=[]
#     for q_i in range(n_trunc):
#         overlap_tot_max_q=[]
#         for q_f in range(n_trunc):
#             overlap_tot_max_q.append(np.array(find_local_maxima(overlap_tot[:,q_i,q_f],index_min=index_min,value_min=value_min)))
#         overlap_tot_max.append(overlap_tot_max_q)

#     overlap_tot_min=[]
#     for q_i in range(n_trunc):
#         overlap_tot_min.append(np.array(find_local_minima(overlap_tot[:,q_i,q_i],index_min=index_min,value_max=value_max)))

#     res_q=[]

#     for q_i in range(n_trunc):
#         overlap_tot_max_q=np.empty((0,3))
#         for q_f in range(n_trunc):
#             if q_i!=q_f:
#                 if overlap_tot_max[q_i][q_f].size!=0:
#                     overlap_tot_max_q=np.vstack([overlap_tot_max_q,np.hstack([overlap_tot_max[q_i][q_f],(q_f*np.ones(len(overlap_tot_max[q_i][q_f]))).reshape(-1,1)])])

#         overlap_tot_max_q_sort=sort_by_nth_column(overlap_tot_max_q,n=0)
#         if overlap_tot_max_q_sort.shape[0]>0:
#             indices_q=find_matching_indices(overlap_tot_min[q_i],overlap_tot_max_q_sort,n_g)
#             for i in range(len(indices_q)):
#                 res_q.append([int(overlap_tot_max_q_sort[indices_q[i][1]][0]),q_i,int(overlap_tot_max_q_sort[indices_q[i][1]][2]),overlap_tot_min[q_i][indices_q[i][0]][1],overlap_tot_max_q_sort[indices_q[i][1]][1]])
#                 #res_q.append([int(overlap_tot_min[q_i][indices_q[i][0]][0]),q_i,int(overlap_tot_max_q_sort[indices_q[i][1]][2]),overlap_tot_min[q_i][indices_q[i][0]][1],overlap_tot_max_q_sort[indices_q[i][1]][1]])

#     # Determine unique rows based on all but the last two columns
#     _, unique_indices = np.unique(np.array(res_q)[:, :-2], axis=0, return_index=True)

#     # Use the unique indices to get the unique rows
#     res_q_unique = np.array(res_q)[sorted(unique_indices)]

#     res_q_sort=sort_by_nth_column(np.array(res_q_unique),n=0)
#     res_q_sort_good=filter_rows(res_q_sort)
#     res_q_sort_clean=remove_rows(res_q_sort_good)

#     swap_statements=[]
#     for i in range(len(res_q_sort_clean)):
#         q0,q1,n=int(res_q_sort_clean[i,1]),int(res_q_sort_clean[i,2]),int(res_q_sort_clean[i,0])
#         swap_statement=True
#         for delta in [-2,-1,1,2]:
#             if f_energies_sorted[q0][n-delta]>f_energies_sorted[q1][n-delta] and f_energies_sorted[q0][n+delta]<f_energies_sorted[q1][n+delta]:
#                 swap_statement=False
#         if swap_statement==True:
#             N_t_sorted_q0=N_t_sorted[q0][n]
#             N_t_sorted_q1=N_t_sorted[q1][n]
#             mean=(N_t_sorted_q0+N_t_sorted_q1)/2
#             # if has_local_extrema(N_t_sorted[q0][n-delta_patch:n+delta_patch]):
#             #     if has_local_extrema(N_t_sorted[q1][n-delta_patch:n+delta_patch]):
#             #         if N_t_sorted_q0<N_t_sorted_q1:
#             #             if np.all(N_t_sorted[q0][n-delta_patch:n+delta_patch]<mean) or np.all(N_t_sorted[q1][n-delta_patch:n+delta_patch]>mean):
#             #                 swap_statement=False
#             #         if N_t_sorted_q0>N_t_sorted_q1:
#             #             if np.all(N_t_sorted[q0][n-delta_patch:n+delta_patch]>mean) or np.all(N_t_sorted[q0][n-delta_patch:n+delta_patch]<mean):
#             #                 swap_statement=False
#         swap_statements.append(swap_statement)

#     index_crit_tot=[]
#     branch_tot=[]
#     gap_tot=[]

#     for q in range(n_trunc):

#         current_q=q
#         index_crit_q=[]
#         branch_q=[]
#         gap_q=[]

#         branch_q.append(current_q)
#         for i in range(len(res_q_sort_clean)):
#             [n,q0,q1]=(res_q_sort_clean[i][0:3]).astype(int)
#             gap=np.abs(f_energies_sorted[q0][n]-f_energies_sorted[q1][n])
#             if q0==current_q:
#                 index_crit_q.append(n)
#                 if swap_statements[i]==True:
#                     current_q=q1
#                     gap_q.append(gap)
#                 else:
#                     gap_q.append(0)
#                 branch_q.append(current_q)
#             elif q1==current_q:
#                 index_crit_q.append(n)
#                 if swap_statements[i]==True:
#                     current_q=q0
#                     gap_q.append(gap)
#                 else:
#                     gap_q.append(0)
#                 branch_q.append(current_q)

#         index_crit_tot.append(index_crit_q)
#         branch_tot.append(branch_q)
#         gap_tot.append(gap_q)


#     N_t_sorted_q_tot=[]
#     f_energies_sorted_q_tot=[]

#     for q in range(n_trunc):
#         N_t_sorted_q=[]
#         f_energies_sorted_q=[]
#         for i in range(len(branch_tot[q])):
#             if len(branch_tot[q])>1:
#                 if i==0:
#                     ind_i=0
#                     ind_f=index_crit_tot[q][i]
#                 elif i==len(branch_tot[q])-1:
#                     ind_i=index_crit_tot[q][i-1]
#                     ind_f=len(overlap_max[q])+1
#                 else:
#                     ind_i=index_crit_tot[q][i-1]
#                     ind_f=index_crit_tot[q][i]
#             else:
#                 ind_i=0
#                 ind_f=len(overlap_max[q])+1

#             N_t_sorted_q.append(N_t_sorted[branch_tot[q][i]][ind_i:ind_f])
#             f_energies_sorted_q.append(f_energies_sorted[branch_tot[q][i]][ind_i:ind_f])

#         # Concatenate the sorted sub-arrays into a single list
#         N_t_sorted_q = [item for list in N_t_sorted_q for item in list]
#         N_t_sorted_q_tot.append(N_t_sorted_q)

#         f_energies_sorted_q = [item for list in f_energies_sorted_q for item in list]
#         f_energies_sorted_q_tot.append(f_energies_sorted_q)

#     return res_q_sort_clean,swap_statements,index_crit_tot,branch_tot,f_energies_sorted_q_tot,N_t_sorted_q_tot,gap_tot


####################################################### FIND RESONANCES WITH OVERLAPS  #############################################################################

# def local_minima(vec,index_min=1,max_value=0.99999):
#     """
#     Function to find minimal values and indexes of the overlap, if overlap > max value
#     """
#     index=[]
#     min=[]
#     for i in range(len(vec)-1):
#         if i>index_min and vec[i]<max_value:
#             if vec[i+1]>vec[i] and vec[i-1]>vec[i]:
#                 index.append(i)
#                 min.append(vec[i])
#     return np.array(index),np.array(min)

# def calculate_score(element1, element2, float_value1, float_value2, q1, q2):
#     """
#     Function to calculate the score to pair up the resonances.
#     element : drive or photon number index
#     float value : overlap value at index
#     q : branch index
#     """
#     score=np.abs(element1 - element2)/100 + np.abs(float_value1 - float_value2)*10
#     if q1==q2:
#         score=1e10 #to avoid pairing up one branch to itself
#     return score

# def find_optimal_pairs(elements,score_threshold=1e-1):
#     """
#     Find optimal pairs by minimizing the score. Pairs are defined only if their score < score_threshold.
#     """
#     pairs = []
#     used = np.zeros(len(elements), dtype=bool)

#     for i in range(len(elements)):
#         if used[i]:
#             continue
#         best_pair = None
#         best_score = float('inf')
#         for j in range(i + 1, len(elements)):
#             if used[j]:
#                 continue
#             item1, q1, _, float_value1 = elements[i]
#             item2, q2, _, float_value2 = elements[j]
#             score = calculate_score(item1, item2, float_value1, float_value2, q1, q2)
#             if score < best_score and score < score_threshold:
#                 best_score = score
#                 best_pair = (i, j)
#         if best_pair:
#             pairs.append(best_pair)
#             used[best_pair[0]] = True
#             used[best_pair[1]] = True
#     return pairs

# def find_resonances_overlap(n_trunc,overlap_max,f_energies_sorted,N_t_sorted,index_min=0,max_value=0.99999):
#     """
#     Find the resonances with overlap metric.

#     Inputs :
#     n_trunc: Transmon hilbert space size in eigenbasis
#     overlap_max: Maximum overlap found by tracking procedure of each branch
#     f_energies_sorted: Floquet energies of each branch
#     N_t_sorted: Transmon average excitation number of each branch
#     """
#     ncrit_index_tot=[]
#     overlap_min_tot=[]
#     for q in range(n_trunc):
#         ncrit_index,overlap_min=local_minima(overlap_max[q],index_min=index_min,max_value=max_value)
#         overlap_min_tot.append(overlap_min)
#         ncrit_index_tot.append(ncrit_index)

#     elements = []
#     for q, (array, float_array) in enumerate(zip(ncrit_index_tot, overlap_min_tot)):
#         for i, (element, float_value) in enumerate(zip(array, float_array)):
#             elements.append((element, q, i, float_value))

#     # Find optimal pairs
#     pairs = find_optimal_pairs(elements)
#     pairs_array=[]
#     for pair in pairs:
#         pairs_array.append([elements[pair[0]][1],elements[pair[1]][1],elements[pair[0]][0]])
#     pairs_array_sorted=sorted(pairs_array, key=lambda x:x[2])

#     swap_statements=[]
#     for i in range(len(pairs_array_sorted)):
#         q0,q1,n=pairs_array_sorted[i]
#         swap_statement=True
#         for delta in [-2,-1,1,2]:
#             if f_energies_sorted[q0][n-delta]>f_energies_sorted[q1][n-delta] and f_energies_sorted[q0][n+delta]<f_energies_sorted[q1][n+delta]:
#                 swap_statement=False
#         swap_statements.append(swap_statement)

#     index_crit_tot=[]
#     branch_tot=[]
#     gap_tot=[]

#     for q in range(n_trunc):
#         current_q=q
#         index_crit_q=[]
#         branch_q=[]
#         gap_q=[]

#         branch_q.append(current_q)
#         for i in range(len(pairs_array_sorted)):
#             q0,q1,n=pairs_array_sorted[i]
#             gap=np.abs(f_energies_sorted[q0][n]-f_energies_sorted[q1][n])
#             if q0==current_q:
#                 index_crit_q.append(n)
#                 if swap_statements[i]==True:
#                     current_q=q1
#                     gap_q.append(gap)
#                 else:
#                     gap_q.append(0)
#                 branch_q.append(current_q)
#             elif q1==current_q:
#                 index_crit_q.append(n)
#                 if swap_statements[i]==True:
#                     current_q=q0
#                     gap_q.append(gap)
#                 else:
#                     gap_q.append(0)
#                 branch_q.append(current_q)

#         index_crit_tot.append(index_crit_q)
#         branch_tot.append(branch_q)
#         gap_tot.append(gap_q)

#     N_t_sorted_q_tot=[]
#     f_energies_sorted_q_tot=[]

#     for q in range(n_trunc):
#         N_t_sorted_q=[]
#         f_energies_sorted_q=[]
#         for i in range(len(branch_tot[q])):
#             if len(branch_tot[q])>1:
#                 if i==0:
#                     ind_i=0
#                     ind_f=index_crit_tot[q][i]
#                 elif i==len(branch_tot[q])-1:
#                     ind_i=index_crit_tot[q][i-1]
#                     ind_f=len(overlap_max[q])+1
#                 else:
#                     ind_i=index_crit_tot[q][i-1]
#                     ind_f=index_crit_tot[q][i]
#             else:
#                 ind_i=0
#                 ind_f=len(overlap_max[q])+1

#             N_t_sorted_q.append(N_t_sorted[branch_tot[q][i]][ind_i:ind_f])
#             f_energies_sorted_q.append(f_energies_sorted[branch_tot[q][i]][ind_i:ind_f])

#         # Concatenate the sorted sub-arrays into a single list
#         N_t_sorted_q = [item for list in N_t_sorted_q for item in list]
#         N_t_sorted_q_tot.append(N_t_sorted_q)

#         f_energies_sorted_q = [item for list in f_energies_sorted_q for item in list]
#         f_energies_sorted_q_tot.append(f_energies_sorted_q)

#     return pairs_array_sorted,swap_statements,index_crit_tot,branch_tot,f_energies_sorted_q_tot,N_t_sorted_q_tot,gap_tot


####################################################### FIND RESONANCES WITH N_T  #############################################################################


def find_resonances_N_t(
    q_array,
    f_modes_0_sorted,
    Transmon_params,
    Floquet_params,
    y_threshold=0.001,
    distance_max=50,
):
    n_trunc = Transmon_params["n_trunc"]
    eps_d_vec = Floquet_params["eps_d_vec"]

    index_der_tot = []
    for q in range(n_trunc):
        N_t_q = N_t(f_modes_0_sorted[q], Transmon_params)
        der_N_t_q = N_t_q[1:] - N_t_q[0:-1]
        der2_N_t_q = N_t_q[2:] - 2 * N_t_q[1:-1] + N_t_q[0:-2]

        index_der_q = []
        for i in range(len(eps_d_vec) - 4):
            if (
                der2_N_t_q[i] > 0
                and der2_N_t_q[i + 1] < 0
                and der2_N_t_q[i - 1] > 0
                and der2_N_t_q[i + 2] < 0
                and np.abs(der_N_t_q[i]) > y_threshold
            ):
                index_der_q.append(i)
            elif (
                der2_N_t_q[i] < 0
                and der2_N_t_q[i + 1] > 0
                and der2_N_t_q[i - 1] < 0
                and der2_N_t_q[i + 2] > 0
                and np.abs(der_N_t_q[i]) > y_threshold
            ):
                index_der_q.append(i)

        index_der_tot.append(index_der_q)

    index_der_tot_array = []
    for i in range(len(index_der_tot)):
        for j in range(len(index_der_tot[i])):
            index_der_tot_array.append([i, index_der_tot[i][j]])

    crossing = []

    for add in range(0, distance_max):
        for i in range(len(index_der_tot_array)):
            q_i = index_der_tot_array[i][0]
            for j in range(len(index_der_tot_array)):
                q_f = index_der_tot_array[j][0]
                if j > i:
                    if (
                        index_der_tot_array[i][1] == index_der_tot_array[j][1] + add
                        and index_der_tot_array[i][1] > 0
                        and index_der_tot_array[j][1] > 0
                    ):
                        crossing.append([q_i, q_f, index_der_tot_array[i][1]])
                        index_der_tot_array[i] = [0, 0]
                        index_der_tot_array[j] = [0, 0]
                    if (
                        index_der_tot_array[i][1] == index_der_tot_array[j][1] - add
                        and index_der_tot_array[i][1] > 0
                        and index_der_tot_array[j][1] > 0
                    ):
                        crossing.append([q_i, q_f, index_der_tot_array[i][1]])
                        index_der_tot_array[i] = [0, 0]
                        index_der_tot_array[j] = [0, 0]

    # crossing = []
    # for q_i in range(n_trunc):
    #    for q_f in range(n_trunc):
    #        if q_f>q_i:
    #            index_match = [x for x in index_der_tot[q_i] if x in index_der_tot[q_f]]
    #            for i_match in index_match:
    #                crossing.append([q_i,q_f,i_match])
    #            index_match = [x for x in index_der_tot[q_i] if x in [y+1 for y in index_der_tot[q_f]]]
    #            for i_match in index_match:
    #                crossing.append([q_i,q_f,i_match])
    #           index_match = [x for x in index_der_tot[q_i] if x in [y-1 for y in index_der_tot[q_f]]]
    #            for i_match in index_match:
    #                crossing.append([q_i,q_f,i_match])
    #            index_match = [x for x in index_der_tot[q_i] if x in [y+2 for y in index_der_tot[q_f]]]
    #            for i_match in index_match:
    #                crossing.append([q_i,q_f,i_match])
    #            index_match = [x for x in index_der_tot[q_i] if x in [y-2 for y in index_der_tot[q_f]]]
    #            for i_match in index_match:
    #                crossing.append([q_i,q_f,i_match])

    crossing = np.array(crossing)

    if len(crossing) == 0:
        q_list_tot = [[0], [1]]
        index_match_list_tot = [[0, len(eps_d_vec)], [0, len(eps_d_vec)]]

    else:
        crossing = crossing[np.argsort(crossing[:, 2])]

        q_list_tot = []
        index_match_list_tot = []
        for q in q_array:
            q_list = []
            q_list.append(q)
            index_match_list = [0]

            for i in range(crossing.shape[0]):
                if q == crossing[i, 0]:
                    q = crossing[i, 1]
                    q_list.append(q)
                    index_match_list.append(crossing[i, 2])
                elif q == crossing[i, 1]:
                    q = crossing[i, 0]
                    q_list.append(q)
                    index_match_list.append(crossing[i, 2])
            index_match_list.append(len(eps_d_vec))

            q_list_tot.append(q_list)
            index_match_list_tot.append(index_match_list)

    return (q_list_tot, index_match_list_tot)


# trouver le gap minimal entre deux
def gap(omega_d, eps_d_vec, index_match_list, q_list, f_energies_sorted):
    gap = []
    for index_match_i, index_match in enumerate(index_match_list[0]):
        if index_match > 0 and index_match < len(eps_d_vec):
            q_i = q_list[0][index_match_i - 1]
            q_f = q_list[0][index_match_i]
            gap_1 = np.abs(
                f_energies_sorted[q_f][index_match]
                - f_energies_sorted[q_i][index_match]
            )
            gap_2 = omega_d - np.abs(
                f_energies_sorted[q_f][index_match]
                - f_energies_sorted[q_i][index_match]
            )
            gap.append(np.min((gap_1, gap_2)))
    return np.array(gap)
