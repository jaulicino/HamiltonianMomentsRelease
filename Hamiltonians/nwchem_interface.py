def get_spin_integrals(yaml_data):
    '''Populate h_one and h_two in spin indexing from NWChem yaml data'''
    n_o = yaml_data['integral_sets'][0]['n_orbitals']
    one_electron_import = (yaml_data['integral_sets'][0]['hamiltonian']
                           ['one_electron_integrals']['values'])
    two_electron_import = (yaml_data['integral_sets'][0]['hamiltonian']
                           ['two_electron_integrals']['values'])
    one_electron_spatial_integrals = np.zeros((n_o, n_o))
    two_electron_spatial_integrals = np.zeros((n_o, n_o, n_o, n_o))
    for val in one_electron_import:
        # Necessary because Python Indexing Starts at 0
        i = int(val[0] - 1)
        j = int(val[1] - 1)
        #
        one_electron_spatial_integrals[i, j] = val[2]
        # Fill the remaining triangular portion
        if i != j:
            one_electron_spatial_integrals[j, i] = val[2]
    # Population the 2-electron spatial Hamiltonian
    for ind, val in enumerate(two_electron_import):
        # Necessary because Python Indexing Starts at 0
        i = int(val[0] - 1)
        j = int(val[1] - 1)
        k = int(val[2] - 1)
        l = int(val[3] - 1)
        #
        two_electron_spatial_integrals[i, j, k, l] = val[4]

        # In most standard Hamiltonians from NWChem, only unique
        # elements are formatted in the Yaml file. The following
        # will fill in the remaining elements. In the case of Ducc Hamiltonians
        # where certain symmetries are lost, all elements are formatted
        # in the Yaml file and this routine will simply read them in
        if two_electron_spatial_integrals[k, l, i, j] == 0:
            two_electron_spatial_integrals[k, l, i, j] = val[4]
        if two_electron_spatial_integrals[i, j, l, k] == 0:
            two_electron_spatial_integrals[i, j, l, k] = val[4]
        if two_electron_spatial_integrals[l, k, i, j] == 0:
            two_electron_spatial_integrals[l, k, i, j] = val[4]
        if two_electron_spatial_integrals[j, i, k, l] == 0:
            two_electron_spatial_integrals[j, i, k, l] = val[4]
        if two_electron_spatial_integrals[k, l, j, i] == 0:
            two_electron_spatial_integrals[k, l, j, i] = val[4]
        if two_electron_spatial_integrals[j, i, l, k] == 0:
            two_electron_spatial_integrals[j, i, l, k] = val[4]
        if two_electron_spatial_integrals[l, k, j, i] == 0:
            two_electron_spatial_integrals[l, k, j, i] = val[4]
    # Construct h_one
    # Alpha_alpha and Beta_beta blocks are equal to the one_electron array
    h_one = np.block(
        [[one_electron_spatial_integrals,
          np.zeros((int(n_o), int(n_o)))],
         [np.zeros((int(n_o), int(n_o))), one_electron_spatial_integrals]])
    # Construct h_two
    h_two = np.zeros((2 * n_o, 2 * n_o, 2 * n_o, 2 * n_o))
    for i in range(len(two_electron_spatial_integrals)):
        for j in range(len(two_electron_spatial_integrals)):
            for k in range(len(two_electron_spatial_integrals)):
                for l in range(len(two_electron_spatial_integrals)):
                    # INDEXING IS 1-(# ALPHA)-(# ALPHA+1)-(# ALPHA+BETA)
                    h_two[i, j, k + n_o,
                          l + n_o] = two_electron_spatial_integrals[i, j, k, l]
                    h_two[i + n_o, j + n_o, k,
                          l] = two_electron_spatial_integrals[i, j, k, l]
                    #
                    if i != k and j != l:
                        h_two[i, j, k,
                              l] = two_electron_spatial_integrals[i, j, k, l]
                        h_two[i + n_o, j + n_o, k + n_o,
                              l + n_o] = (two_electron_spatial_integrals[i, j,
                                                                         k, l])
    return h_one, 0.5 * h_two


def yaml_read(nw_data_file):
    '''Read in NWChem Yaml file and output Hamiltonian json file.'''
    import yaml
    from yaml import SafeLoader
    from qiskit.chemistry import FermionicOperator
    with open(nw_data_file, 'r') as file:
        data = yaml.load(file, SafeLoader)
    fop = FermionicOperator(get_spin_integrals(data))
    obj_p = fop.mapping('jordan_wigner')
    jw_dict = obj_p.to_dict()['paulis']
    Ld = len(jw_dict)
    ham_one = {}
    for i in range(0, Ld):
        key = jw_dict[i]['label']
        ham_one[key] = str(jw_dict[i]['coeff']['real'] +
                           1j * jw_dict[i]['coeff']['imag'])
    #
    # Auxiliary calcualtions
    # from qiskit.aqua.algorithms import NumPyEigensolver
    # nuclear_repulsion_energy = data['integral_sets'][0]['coulomb_repulsion']['value']
    # exact_eigensolver = NumPyEigensolver(obj_p, k=1)
    # ret = exact_eigensolver.run()
    # print('THE ELECTRONIC ENERGY = {:.12f}'.format(ret['eigenvalues'][0].real))
    # print('NUCLEAR REPULSION ENERGY = {:.12f}'.format(nuclear_repulsion_energy))
    # print('TOTAL FCI ENERGY = " {:.12f}'.format(ret['eigenvalues'][0].real +
    # nuclear_repulsion_energy))
    #
    return ham_one