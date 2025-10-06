import numpy as np


def generate_branches(evals, evecs, seed_indices, size_branch, operator, break_tie=True):
    """
    Starting from a set of eigenvalues and eigenvectors, and starting from a set of specified seed eigenvectors, generates
    sequences of eigenvalues and eigenvectors of length branch_size such that consecutive eigenvectors in the sequence
    are most strongly coupled by the specified operator. Each sequence is called a branch. The branches are built in parallel
    from each seed. By default, each branch is assigned a unique set of eigenvectors without duplication. However, duplicate
    assignment can be allowed as an option.

    The metric that is maximized to determine the most strongly coupled eigenvectors is the one proposed in
    Shillito et al., Phys. Rev. Applied 18, 034031 (2022). Let |v0> be the initial eigenvector, let |v1> be the final eigenvector,
    and let op be the specified operator. Then the metric used to find the most stronlgy coupled |v1> is:

    metric = |<v1|op|v0>|**2 / < v0|op.dag*op|v0>

    Parameters
    ----------

    evals : 1D real ndarray
        A one-dimensional real ndarray containing the eigenvalues to choose from.

    evecs : 2D complex ndarray
        A two-dimensional complex ndarray containing the eigenvectors to choose from, with each column containing an eigenvector.

    seed_indices : 1D integer ndarray
        The indices of the eigenvalues and eigenvectors to use as seeds for branch generation.

    size_branch : Positive integer
        The number of eigenvalues and eigenvectors to include in each branch.

    operator : 2D complex square ndarray
        The operator whose matrix element is used to assign the next eigenstate within a branch. It must have the same dimensions as the eigenvectors.

    break_tie : Boolean
        Whether to break the tie between two branches or allow duplicate assignment. The default value is True.

    
    Returns
    -------

    branches_evals : 2D real ndarray
        A two-dimensional real ndarray containing the eigenvalues of each branch. The first dimension indexes the branches and the second dimension indexes the eigenvalues.

    branches_evecs : 3D complex ndarray
        A three-dimensional complex ndarray containing the eigenvectors of each branch. The first dimension indexes the branches, such that branches_evecs[k] is an array
        with each column containing an eigenvector of the branch.

    branches_indices : 2D integer ndarray
        A two-dimensional integer ndarray containing the indices of the eigenvectors of each branch. The first dimension indexes the branches and the second dimension indexes the indices.

    branches_metrics : 2D real ndarray
        A two-dimensional real ndarray containing the optimum assignment metric for each branch. The first dimension indexes the branches and the second dimension indexes the metrics.

    """

    # Number of branches and size of the eigenvectors
    num_branches = seed_indices.shape[0]
    size_evecs = evecs.shape[0]

    # Initialize the branch properties
    branches_evals = np.zeros((num_branches, size_branch))
    branches_evecs = np.zeros((num_branches, size_evecs, size_branch), dtype=np.cdouble)
    branches_indices = np.zeros((num_branches, size_branch), dtype=np.uint)
    branches_metrics = np.zeros((num_branches, size_branch-1))

    # Extract the seed eigenvalues and eigenvectors
    seed_evals = evals[seed_indices]
    seed_evecs = evecs[:,seed_indices]

    # Extract the target eigenvalues, eigenvectors, and indices
    target_evals = np.delete(evals, seed_indices, axis=0)
    target_evecs = np.delete(evecs, seed_indices, axis=1)
    target_indices = np.delete(np.arange(evecs.shape[1], dtype=np.uint), seed_indices, axis=0)

    # Store the initial branch properties
    branches_evals[:,0] = seed_evals
    branches_evecs[:,:,0] = seed_evecs.T
    branches_indices[:,0] = seed_indices

    # For all branch increments...
    for k in range(1,size_branch):

        # Calculate the branch metrics
        metrics = calculate_metrics(seed_evecs, target_evecs, operator)

        # Find the optimum indices and metrics within the calculated metrics
        opt_idxs, opt_metrics = find_optimum_indices(metrics, break_tie)

        # Update the seed eigenvalues, eigenvectors, and indices
        seed_evals = target_evals[opt_idxs]
        seed_evecs = target_evecs[:,opt_idxs]
        seed_indices = target_indices[opt_idxs]
    
        # Update the target eigenvalues, eigenvectors, and indices
        target_evals = np.delete(target_evals, opt_idxs, axis=0)
        target_evecs = np.delete(target_evecs, opt_idxs, axis=1)
        target_indices = np.delete(target_indices, opt_idxs, axis=0)

        # Store the updated eigenvalues and eigenvectors
        branches_evals[:,k] = seed_evals
        branches_evecs[:,:,k] = seed_evecs.T
        branches_indices[:,k] = seed_indices
        branches_metrics[:,k-1] = opt_metrics

    return branches_evals, branches_evecs, branches_indices, branches_metrics


def calculate_metrics(seed_evecs, target_evecs, operator):

    """
    Calculate the metric determining how strongly the seed and target eigenvectors are coupled by the specified operator.
    This is the metric proposed in Shillito et al., Phys. Rev. Applied 18, 034031 (2022). Let |v0> be the seed eigenvector, let |v1> be
    the target eigenvector, and let op be the specified operator. Then the coupling metric between |v0> and |v1> is

    metric = |<v1|op|v0>|**2 / < v0|op.dag*op|v0>

    Parameters
    ----------

    seed_evecs : 2D complex ndarray
        Two-dimensional complex ndarray containing the eigenvectors to use as seeds, with each column containing an eigenvector.

    target_evecs : 2D complex ndarray
        Two-dimensional complex ndarray containing the eigenvectors to use as targets, with each column containing an eigenvector.

    operator : 2D complex square ndarray
        The operator coupling the seed and target. It must have the same dimensions as the eigenvectors.

    Returns
    -------

    metrics : 2D real ndarray
        Two-dimensional read ndarray containing the metrics determining how strongly the seed eigenvectors are coupled to the target eigenvectors.
        The first dimension indexes the seeds and the second dimension indexes the targets.

    """

    # Calculate the metrics between the seed and target eigenvectors
    operated_seed_evecs = operator @ seed_evecs
    # Calculate the square of the matrix elements
    numerator = np.abs( target_evecs.conj().T @ operated_seed_evecs ) ** 2
    # Calculate the normalization
    denominator = np.linalg.norm(operated_seed_evecs, axis=0) ** 2
    # Calculate the metrics
    metrics = numerator / denominator
    metrics = metrics.T

    return metrics


def find_optimum_indices(metrics, break_tie):
    """
    Starting from an array of metrics between seeds and targets, find the target indices such that metric is maximized for each seed. If break_tie
    is set to True, the seeds with the highest target metrics are given priority and there is no duplicate assignment. If break_tie is set to False,
    each seed is assigned a target independently based on its the maximum target metric, and duplicates can occur.

    Parameters
    ----------

    metrics : 2D real ndarray
        Two-dimensional ndarray containing the metrics between seeds and targets. The first dimension indexes the seeds and the second dimension indexes the targets.

    break_tie : Boolean
        Whether to break the tie between two seeds or allow duplicate assignment.

    Returns
    -------

    opt_metrics_indices : 1D integer ndarray
        One-dimensional ndarray containing the indices of the optimal metrics for each seed.

    opt_metrics : 1D real ndarray
        One-dimensional ndarray containing the optimal metrics for each seed.

    """

    # If a tie break is required ...
    if break_tie is True:

        # Extract the number of seeds
        num_seeds = metrics.shape[0]

        # For each seed, partition the metrics so that the first num leading (largest) elements are at the beginning of the list, and then keep only these elements.
        # For large lists, this is usually faster than sorting through the whole list.
        leading_metrics_indices = np.argpartition(-metrics, num_seeds-1, axis=1)
        leading_metrics_indices = leading_metrics_indices[:,0:num_seeds]
        leading_metrics = np.take_along_axis(metrics, leading_metrics_indices, axis=1)

        # Sort the partitioned leading metrics
        sort_idxs = np.argsort(-leading_metrics, axis=1)
        leading_metrics_indices = np.take_along_axis(leading_metrics_indices, sort_idxs, axis=1)
        leading_metrics = np.take_along_axis(leading_metrics, sort_idxs, axis=1)

        # Initialize the optimal metric indices and metrics
        opt_metrics_indices = np.zeros((num_seeds,))
        opt_metrics = np.zeros((num_seeds,))

        # For all seeds ...
        for k in range(num_seeds):

            # Find the rows and column indices where the remaining largest metrics are
            rows_rem_seeds, cols_rem_largest_metrics = np.ma.notmasked_edges(leading_metrics_indices, axis=1)[0]
            
            # Find the row and column that has the winning index
            max_idx = np.argmax(leading_metrics[rows_rem_seeds, cols_rem_largest_metrics])
            opt_row = rows_rem_seeds[max_idx]
            opt_col = cols_rem_largest_metrics[max_idx]

            # Find the winning index and store it
            opt_metric_index = leading_metrics_indices[opt_row, opt_col]
            opt_metric = leading_metrics[opt_row, opt_col]
            opt_metrics_indices[opt_row] = opt_metric_index
            opt_metrics[opt_row] = opt_metric

            # Mask the winning seed so that it cannot be used again
            mask = np.zeros(leading_metrics_indices.shape, dtype=bool)
            mask[opt_row,:] = True
            leading_metrics_indices = np.ma.masked_array(leading_metrics_indices, mask=mask)
            # Mask the chosen index so that it cannot be picked by the remaining seeds
            leading_metrics_indices = np.ma.masked_equal(leading_metrics_indices, opt_metric_index)

    # If a tie break is not required...
    else:

        # Independently find the index with the maximum metric for each seed
        opt_metrics_indices = np.argmax(metrics, axis=1, keepdims=True)
        opt_metrics = np.take_along_axis(metrics, opt_metrics_indices, axis=1)
        # Remove trailing dimension
        opt_metrics_indices = opt_metrics_indices[..., 0]
        opt_metrics = opt_metrics[..., 0]

    # Convert the optimal metric indices to integer types
    opt_metrics_indices = opt_metrics_indices.astype(np.uint)

    return opt_metrics_indices, opt_metrics



