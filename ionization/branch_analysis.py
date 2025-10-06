import numpy as np
import scipy as sp
import qutip as qt


# Function to identify the dressed states i,n
def generate_BA_qutip(eigv_coupl,eigs_coupl,max_fock,n_trunc):
    i_0=[]
    for qubit in range(n_trunc):
        eigv_q0_bare=qt.tensor(qt.basis(n_trunc,qubit),qt.basis(max_fock,0))
        c=[]
        for i in range(int(n_trunc*max_fock)):
            c.append(np.abs((eigv_q0_bare.overlap(qt.Qobj(eigv_coupl[i])))))
        i_0.append(np.array(c).argmax())
    i_0=np.array(i_0)

    at=qt.create(max_fock)
    at_op=qt.tensor(qt.qeye(n_trunc),at)
    number_op=qt.tensor(qt.qeye(n_trunc),at*at.dag())
    
    branches_evals, branches_evecs, branches_indices, branches_metrics=generate_branches(eigs_coupl, evecs_qutip_to_numpy(eigv_coupl), i_0, max_fock, at_op.full())
    
    ind_tot=[]
    for i in range(n_trunc):
        ind_tot.append(branches_indices[i,:])

    n_photon_tot=[]
    n_trans_tot=[]
    vec_i=np.arange(0,n_trunc)
    
    for j in range(n_trunc):
        n_photon=[]
        n_trans=[]
        
        for i in ind_tot[j]:
            n_photon.append(np.abs((eigv_coupl[i].dag()*number_op*eigv_coupl[i])[0][0][0]))
            n_trans.append(((eigv_coupl[i].ptrace(0)).diag()).dot(vec_i))
        
        n_photon_tot.append(np.real(np.array(n_photon)))
        n_trans_tot.append(np.real(np.array(n_trans)))

    return ind_tot,n_photon_tot,n_trans_tot

    
# Function to convert numpy eigenvector arrays to qutip eigenvector arrays
def evecs_numpy_to_qutip(evecs_numpy, dims=None):
    evecs_qutip = np.hsplit(evecs_numpy, evecs_numpy.shape[1])
    evecs_qutip = [qt.Qobj(x, dims=dims) for x in evecs_qutip]
    return evecs_qutip

# Function to convert qutip eigenvector arrayst to numpy eigenvector arrays
def evecs_qutip_to_numpy(evecs_qutip):
    evecs_numpy = np.block([x.full() for x in evecs_qutip])
    return evecs_numpy



# Function to identify the dressed states i,n
def generate_BA(eigv_coupl,eigs_coupl,max_fock,n_trunc,omega_r=0,method='overlap',delta_fac=1e-2):
    num_op_t=np.diag(np.arange(0,n_trunc))
    I_op_t=np.eye(n_trunc)
    
    num_op_r=np.diag(np.arange(0,max_fock))
    I_op_r=np.eye(max_fock)
    
    num_op_t_tot=np.kron(num_op_t,I_op_r)
    num_op_r_tot=np.kron(I_op_t,num_op_r)
    
    n_trans_array=np.diag(np.dot(np.dot(eigv_coupl.conj().T,num_op_t_tot),eigv_coupl))
    n_photon_array=np.diag(np.dot(np.dot(eigv_coupl.conj().T,num_op_r_tot),eigv_coupl))
    
    i_0=[]
    for qubit in range(n_trunc):
        qubit_vec=np.zeros(n_trunc)
        qubit_vec[qubit]=1
        res_vec=np.zeros(max_fock)
        res_vec[0]=1
        eigv_q0_bare=np.kron(qubit_vec,res_vec)
        i_0.append(np.argmax(np.abs(np.dot(eigv_q0_bare.conj().T,eigv_coupl))))
    i_0=np.array(i_0)
    
    at_op=np.diag(np.sqrt(np.arange(1,max_fock)),-1)
    at_op_tot=np.kron(I_op_t,at_op)
    
    if method=='overlap':
        branches_evals, branches_evecs, branches_indices, branches_metrics=generate_branches(eigs_coupl, eigv_coupl, i_0, max_fock, at_op_tot)
    elif method=='goto':
        branches_evals, branches_evecs, branches_indices = generate_branches_goto(eigs_coupl, eigv_coupl, num_op_t_tot, omega_r, i_0, max_fock,n_trunc,delta_fac)
        branches_metrics=None
    else: 
        print('ERROR: method must be equal to "overlap" or "goto". ')
    
    ind_tot=[]
    for i in range(n_trunc):
        ind_tot.append(branches_indices[i,:])
    
    n_photon_tot=[]
    n_trans_tot=[]
    for j in range(n_trunc):
        n_photon_tot.append(np.real(np.array(n_photon_array[ind_tot[j]])))
        n_trans_tot.append(np.real(np.array(n_trans_array[ind_tot[j]])))

    
    I_op_t=np.eye(n_trunc)
    at_op=np.diag(np.sqrt(np.arange(1,max_fock)),-1)
    at_op_tot=np.kron(I_op_t,at_op)
        
    comparaison=at_op_tot@eigv_coupl
    c_tot=eigv_coupl.T.conj()@comparaison
    
    fac_norm=np.diag((comparaison.T.conj())@comparaison)
    fac_norm_array = np.tile(fac_norm, (len(fac_norm), 1))
    c_tot_norm=np.abs(c_tot/fac_norm_array)**2

    return ind_tot,n_photon_tot,n_trans_tot,branches_metrics,c_tot_norm



####################################################### XANDER METHOD #############################################################################

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
    denominator = sp.linalg.norm(operated_seed_evecs, axis=0) ** 2
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

    
####################################################### FIND RESONANCES WITH OVERLAPS #############################################################################


def map_matrix_to_coordinates(matrix):
    n = len(matrix)
    m = len(matrix[0])
    
    # Create the output array
    output = [None] * (n * m)
    
    # Iterate through the matrix and fill the output array
    for i in range(n):
        for j in range(m):
            k = matrix[i][j]
            output[k] = (i, j)

    return np.array(output)
def find_indices_greater_than_threshold(matrix, threshold):
    indices = []
    overlaps = []
    for x in range(len(matrix)):
        for y in range(len(matrix[0])):
            if matrix[x][y] > threshold and [y,x] not in indices :
                indices.append([x, y])
                overlaps.append( matrix[x][y])
    return np.array(indices),np.array(overlaps)
def sort_matrix(matrix):
    a, b = matrix[0]
    c, d = matrix[1]
    if a < c:
        return np.array([[a, b], [c, d]])
    else:
        return np.array([[c, d], [a, b]])

def find_resonances_overlap(c_tot,ind,n_trans,n_photon,eigs_coupl,max_fock,n_trunc,omega_r,threshold=1e-6):
    output=map_matrix_to_coordinates(np.array(ind))
    indices,overlaps=find_indices_greater_than_threshold(c_tot,threshold)
    indices_q_n=output[list(indices[:,:2])]
    indices_q_n_sort=[]
    overlaps_sort=[]
    for i in range(len(indices_q_n)):
        if indices_q_n[i,0,0]!=indices_q_n[i,1,0] and indices_q_n[i,0,1]<max_fock-10 and indices_q_n[i,1,1]<max_fock-10:
            indices_q_n_sort.append(indices_q_n[i])
            overlaps_sort.append(overlaps[i])
    indices_q_n_sort=np.array(indices_q_n_sort)
    overlaps_sort=np.array(overlaps_sort)
    indices_q_n_sorted = np.array([sort_matrix(matrix) for matrix in indices_q_n_sort])

    indices_q_n_sorted_final = np.array(sorted(indices_q_n_sorted, key=lambda x: (x[0, 0],x[0, 1])))
    sorting_indices = np.lexsort((indices_q_n_sorted[:, 0, 1], indices_q_n_sorted[:, 0, 0]))
    overlaps_sorted_final = overlaps_sort[sorting_indices]
    for i in np.flip(np.arange(0,len(indices_q_n_sorted_final))):
        if indices_q_n_sorted_final[i,0,1]==indices_q_n_sorted_final[i,1,1] and indices_q_n_sorted_final[i,0,0]==indices_q_n_sorted_final[i,1,0]-1:
            indices_q_n_sorted_final=np.delete(indices_q_n_sorted_final,i,axis=0)
            overlaps_sorted_final=np.delete(overlaps_sorted_final,i)
   
    
    # Group matrices based on resonances between q0 and q1
    groups = {}
    overlap_res = {}
    i=0
    for matrix in indices_q_n_sorted_final:
        key = tuple(matrix[:, 0])
        if key not in groups:
            groups[key] = []
            overlap_res[key] = []
        groups[key].append(matrix)
        overlap_res[key].append(overlaps_sorted_final[i])
        i+=1
    result_indices = [np.array(group) for group in groups.values()]
    result_overlaps = [np.array(group) for group in overlap_res.values()]
    
    # Group matrices based on different resonances between q0 and q1 at different photon numbers
    subgroups_indices = []
    subgroups_overlaps = []
    for i in range(len(result_indices)):    
        sorted_array_indices = result_indices[i]
        sorted_array_overlaps = result_overlaps[i]
        current_subgroup_indices = [sorted_array_indices[0]]
        current_subgroup_overlaps = [sorted_array_overlaps[0]]
        index=0
        for matrix in sorted_array_indices[1:]:
            if abs(matrix[0, 1] - current_subgroup_indices[-1][0, 1]) <= 1:
                current_subgroup_indices.append(matrix)
                current_subgroup_overlaps.append(sorted_array_overlaps[index])
            else:
                subgroups_indices.append(np.array(current_subgroup_indices))
                subgroups_overlaps.append(np.array(current_subgroup_overlaps))
                current_subgroup_indices = [matrix]
                current_subgroup_overlaps = [sorted_array_overlaps[index]]
            index+=1  
        subgroups_indices.append(np.array(current_subgroup_indices))
        subgroups_overlaps.append(np.array(current_subgroup_overlaps))
    
    # Determine if branches swap at the resonance
    subgroups_swaps=[]
    for res in range(len(subgroups_indices)):
        subgroups_indices_i=subgroups_indices[res][0]
        subgroups_indices_f=subgroups_indices[res][-1]
        q0_i,n0_i=subgroups_indices_i[0]
        q1_i,n1_i=subgroups_indices_i[1]  
        q0_f,n0_f=subgroups_indices_f[0]
        q1_f,n1_f=subgroups_indices_f[1]
        n_trans_0i=n_trans[q0_i][n0_i]
        n_trans_1i=n_trans[q1_i][n1_i]
        n_trans_0f=n_trans[q0_f][n0_f]
        n_trans_1f=n_trans[q1_f][n1_f] 
        prob_swap=np.abs(n_trans_1f-n_trans_1i)+np.abs(n_trans_0f-n_trans_0i)
        prob_noswap=np.abs(n_trans_1f-n_trans_0i)+np.abs(n_trans_0f-n_trans_1i) 
        if prob_swap>prob_noswap:
            subgroups_swaps.append(True)
        else :
            subgroups_swaps.append(False)
        
    # Find the maximum of the overlap. will be use to define ncrit
    maximum_overlaps=[]
    maximum_indices=[]
    for i in range(len(subgroups_indices)):
        index=np.argmax(subgroups_overlaps[i])
        maximum_overlaps.append((subgroups_overlaps[i])[index])
        maximum_indices.append((subgroups_indices[i])[index])


    # Track qubit like state 
    maximum_indices_nsort_tot=[]
    subgroups_swaps_nsort_tot=[]
    maximum_overlaps_nsort_tot=[]
    ncrit_overlap_tot=[]

    # Find the minimum gap.
    gap=[]
    gap_indices=[]
    eigs_coupl_mod=eigs_coupl%omega_r
    for i in range(len(subgroups_indices)):
        subgroups_indices_i=subgroups_indices[i]
        gap_q=[]
        for j in range(len(subgroups_indices_i)):
            [[q0,n0],[q1,n1]]=subgroups_indices_i[j]
            gap_q.append(np.abs(eigs_coupl_mod[ind[q0][n0]]-eigs_coupl_mod[ind[q1][n1]]))
        index=np.argmin(np.array(gap_q))
        gap.append((np.array(gap_q))[index])
        gap_indices.append(subgroups_indices_i[index])

        
    # resonances 
    gap_tot=[]
    ncrit_gap_tot=[]
    indices_gap_tot=[]
    maximum_indices_nsort=np.array(sorted(maximum_indices, key=lambda x: (x[0, 1],x[1, 1])))
    sorting_indices = np.lexsort((np.array(maximum_indices)[:, 1, 1], np.array(maximum_indices)[:, 0,1]))
    subgroups_swaps_nsort=np.array(subgroups_swaps)[sorting_indices]
    maximum_overlaps_nsort=np.array(maximum_overlaps)[sorting_indices]
    gap_nsort=np.array(gap)[sorting_indices]
    gap_indices_nsort=np.array(gap_indices)[sorting_indices]
    branch_character=np.arange(0,n_trunc)
    for q in range(n_trunc):
        maximum_indices_nsort_tot.append([])
        subgroups_swaps_nsort_tot.append([])
        maximum_overlaps_nsort_tot.append([])
        ncrit_overlap_tot.append([])
        ncrit_gap_tot.append([])
        gap_tot.append([])
        indices_gap_tot.append([])
    for i in range(len(maximum_indices_nsort)):
        res=maximum_indices_nsort[i]
        q0,n0=res[0]
        q1,n1=res[1]
        branch_character_0=branch_character[q0]
        branch_character_1=branch_character[q1]
        if branch_character[q0]!=branch_character[q1]+1 and branch_character[q0]!=branch_character[q1]-1:
            j=0
            for [q,n] in [[q0,n0],[q1,n1]]:
                branch_character_q=branch_character[q]
                maximum_indices_nsort_tot[branch_character_q].append(res)
                subgroups_swaps_nsort_tot[branch_character_q].append(subgroups_swaps_nsort[i])
                maximum_overlaps_nsort_tot[branch_character_q].append(maximum_overlaps_nsort[i])
                ncrit_overlap_tot[branch_character_q].append(np.array(n_photon)[q,n])
                if subgroups_swaps_nsort[i]==False:
                    gap_tot[branch_character_q].append(0)
                elif subgroups_swaps_nsort[i]==True:
                    gap_tot[branch_character_q].append(gap_nsort[i])
                ncrit_gap_tot[branch_character_q].append(np.array(n_photon)[gap_indices_nsort[i][j,0],gap_indices_nsort[i][j,1]])
                indices_gap_tot[branch_character_q].append(gap_indices_nsort[i])
                j+=1
            if subgroups_swaps_nsort[i]==True:
                branch_character[q0],branch_character[q1]=branch_character[q1],branch_character[q0]
    
                
    # #Track qubit like state 
    # maximum_indices_nsort_tot=[]
    # subgroups_swaps_nsort_tot=[]
    # maximum_overlaps_nsort_tot=[]
    # ncrit_tot=[]
    # maximum_indices_nsort=np.array(sorted(maximum_indices, key=lambda x: (x[0, 1],x[1, 1])))
    # sorting_indices = np.lexsort((np.array(maximum_indices)[:, 1, 1], np.array(maximum_indices)[:, 0,1]))
    # subgroups_swaps_nsort=np.array(subgroups_swaps)[sorting_indices]
    # maximum_overlaps_nsort=np.array(maximum_overlaps)[sorting_indices]
    # branch_character=np.arange(0,n_trunc)
    # for q in range(n_trunc):
    #     maximum_indices_nsort_tot.append([])
    #     subgroups_swaps_nsort_tot.append([])
    #     maximum_overlaps_nsort_tot.append([])
    #     ncrit_tot.append([])  
    # for i in range(len(maximum_indices_nsort)):
    #     res=maximum_indices_nsort[i]
    #     q0,n0=res[0]
    #     q1,n1=res[1]
    #     branch_character_0=branch_character[q0]
    #     branch_character_1=branch_character[q1]
    #     if branch_character[q0]!=branch_character[q1]+1 and branch_character[q0]!=branch_character[q1]-1:
    #         for [q,n] in [[q0,n0],[q1,n1]]:
    #             branch_character_q=branch_character[q]
    #             maximum_indices_nsort_tot[branch_character_q].append(res)
    #             subgroups_swaps_nsort_tot[branch_character_q].append(subgroups_swaps_nsort[i])
    #             maximum_overlaps_nsort_tot[branch_character_q].append(maximum_overlaps_nsort[i])
    #             ncrit_tot[branch_character_q].append(np.array(n_photon)[q0,n0])
    
    #         if subgroups_swaps_nsort[i]==True:
    #             branch_character[q0],branch_character[q1]=branch_character[q1],branch_character[q0]
        
    return subgroups_swaps_nsort_tot,maximum_indices_nsort_tot,maximum_overlaps_nsort_tot,ncrit_overlap_tot,indices_gap_tot,gap_tot,ncrit_gap_tot


####################################################### FIND RESONANCES WITH N_T  #############################################################################

def find_resonances_N_t(q_array,N_t_tot,N_r_tot,Transmon_params,y_threshold=0.001,distance_max=50):
    trunc_charge=Transmon_params['trunc_charge']
    max_fock=Transmon_params['max_fock']
    
    index_der_tot=[]
    for q in range(trunc_charge):
        N_t_q=N_t_tot[q]
        der_N_t_q=N_t_q[1:]-N_t_q[0:-1]
        der2_N_t_q=N_t_q[2:]-2*N_t_q[1:-1]+N_t_q[0:-2]
    
        index_der_q=[]
        for i in range(max_fock-4):
            if der2_N_t_q[i]>0 and der2_N_t_q[i+1]<0 and der2_N_t_q[i-1]>0 and der2_N_t_q[i+2]<0 and np.abs(der_N_t_q[i])>y_threshold:
                index_der_q.append(N_r_tot[q,i])
            elif der2_N_t_q[i]<0 and der2_N_t_q[i+1]>0 and der2_N_t_q[i-1]<0 and der2_N_t_q[i+2]>0 and np.abs(der_N_t_q[i])>y_threshold:
                index_der_q.append(N_r_tot[q,i])
        
        index_der_tot.append(index_der_q)

    index_der_tot_array=[]
    for i in range(len(index_der_tot)):
        for j in range(len(index_der_tot[i])):
            index_der_tot_array.append([i,index_der_tot[i][j]])
            
    crossing=[]
    for add in range(0,distance_max):
        for i in range(len(index_der_tot_array)):
            q_i=index_der_tot_array[i][0]
            for j in range(len(index_der_tot_array)):
                q_f=index_der_tot_array[j][0]
                if j>i:
                    if np.round(index_der_tot_array[i][1])==np.round(index_der_tot_array[j][1]+add) and index_der_tot_array[i][1]>0 and index_der_tot_array[j][1]>0:
                        crossing.append([q_i,q_f,index_der_tot_array[i][1]])
                        index_der_tot_array[i]=[0,0]
                        index_der_tot_array[j]=[0,0]
                    if np.round(index_der_tot_array[i][1])==np.round(index_der_tot_array[j][1]-add) and index_der_tot_array[i][1]>0 and index_der_tot_array[j][1]>0:
                        crossing.append([q_i,q_f,index_der_tot_array[i][1]])
                        index_der_tot_array[i]=[0,0]
                        index_der_tot_array[j]=[0,0]
                    
    #crossing = []
    #for q_i in range(trunc_charge):
    #    for q_f in range(trunc_charge):
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
    
    if len(crossing)==0:
        q_list_tot=[[0],[1]]
        index_match_list_tot=[[0,max_fock],[0,max_fock]]

    else:
        crossing = crossing[np.argsort(crossing[:,2])] 
        
        q_list_tot=[]
        index_match_list_tot=[]
        for q in q_array:
            q_list=[]
            q_list.append(q)
            index_match_list=[0]
            
            for i in range(crossing.shape[0]):
                if q == crossing[i,0]:
                    q = crossing[i,1]
                    q_list.append(q)
                    index_match_list.append(crossing[i,2])
                elif q == crossing[i,1]:
                    q = crossing[i,0]
                    q_list.append(q)
                    index_match_list.append(crossing[i,2])
            index_match_list.append(max_fock)
        
            q_list_tot.append(q_list)
            index_match_list_tot.append(index_match_list)
        
    return(q_list_tot,index_match_list_tot)




####################################################### GOTO METHOD #############################################################################


def expectation_value_N(state, N_operator):
    return np.real(np.dot(state.conj().T, np.dot(N_operator, state)))

def find_closest_eigenstates(candidate_energy, eigenenergies, delta):
    diff = np.abs(eigenenergies - candidate_energy)
    within_window = np.where(diff <= delta / 2)[0]
    if len(within_window) > 0:
        return within_window
    else:
        return np.argsort(diff)[:2]

def generate_branches_goto(eigenenergies, eigenstates, N_operator, omega, initial_indices, max_fock, n_trunc, delta_fac=1e-2):
    delta = delta_fac * omega
    indices = []
    eigs_sorted = []
    eigv_sorted = []
    eigenenergies_available = np.copy(eigenenergies)
    
    for initial_index in initial_indices[:n_trunc]:
        
        current_index = initial_index
        current_eigenenergy = eigenenergies[current_index]
        current_state = eigenstates[:, current_index]

        current_branch = [current_index]
        current_eigs = [current_eigenenergy]
        current_eigv = [current_state]

        for _ in range(max_fock):
            eigenenergies_available[current_index]=-omega*1e10

            # Estimate candidate energy for the next state
            candidate_energy = current_eigenenergy + omega

            # Find eigenstates within the energy window
            candidate_indices = find_closest_eigenstates(candidate_energy, eigenenergies_available, delta)

            # Evaluate the expectation values of N for the selected eigenstates
            candidate_expectations = np.array([expectation_value_N(eigenstates[:, idx], N_operator) for idx in candidate_indices])

            # Get the expectation value of N for the current state
            current_expectation = expectation_value_N(current_state, N_operator)

            # Determine the next state with the closest expectation value to the current state
            closest_index = candidate_indices[np.argmin(np.abs(candidate_expectations - current_expectation))]

            # Update current state information
            current_index = closest_index
            current_eigenenergy = eigenenergies[current_index]
            current_state = eigenstates[:, current_index]

            # Append current state to the branch
            current_branch.append(current_index)
            current_eigs.append(current_eigenenergy)
            current_eigv.append(current_state)
            
        indices.append(current_branch)
        eigs_sorted.append(current_eigs)
        eigv_sorted.append(current_eigv)

    return np.array(eigs_sorted), np.array(eigv_sorted), np.array(indices)
    
def find_resonances_goto(n_photon,n_trans,eigs,ind,omega_r,q_max=2,bound=10,d=5):
    ncrit=[]
    ncrit_index=[]
    gap=[]
    for q in range(q_max):
        n_trans_q=n_trans[q]
        n_photon_q=n_photon[q]
        ncrit_index_q = []
        gap_q=[]
        n = len(n_trans_q)
        
        for i in range(1, n - 1-bound):
            if n_trans_q[i] > n_trans_q[i - 1] and n_trans_q[i] > n_trans_q[i + 1]:
                ncrit_index_q.append(i)
                gap_q.append(np.max(np.abs(eigs[ind[q][i-d:i+d]]-eigs[ind[q][i-d-1:i+d-1]]-omega_r)))
        
        ncrit_q=n_photon_q[ncrit_index_q]
        ncrit.append(ncrit_q)
        ncrit_index.append(ncrit_index_q)
        gap.append(np.array(gap_q))
        
    return ncrit,ncrit_index,gap
