import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix, linalg as sla
from math import pi

sigma_0 = np.array([[1,0],[0,1]])
sigma_x = np.array([[0,1],[1,0]])
sigma_y = np.array([[0,-1j],[1j,0]])
sigma_z = np.array([[1,0],[0,-1]])

def vector_potential(Nx,Ny,fluxes, gauge = "C4"):
    '''Returns Peierls phases of a rectangular square lattice of dimensions (Nx,Ny) given an array of fluxes through each unit cell.
       - Nx: int
       - Ny: int
       - fluxes: ndarray(Nx, Ny)
       - gauge: "Landau" or "C4"
        Returns:
       - a_e: ndarray(Nx,Ny+1)
       - a_n: ndarray(Nx+1,Ny)'''

    def index(direction, i,j):
        '''Returns the index of a Peierls phase in order: direction, xdir, ydir.'''
        if i < Nx and j < Ny:              # first all the hoppings in the unit cell
            idx = (j*Nx+i)*2 + direction 
        elif j == Ny:                       # then all hoppings on the top edge
            idx = 2*Nx*Ny+i
        elif i == Nx:                       # then all hoppings on the right edge
            idx = 2*Nx*Ny+Nx+j
        return idx
    #There are 2*Nx*Ny+Nx+Ny unknowns (all possible values of idx)

    
    row = []
    col = []
    data = []
    rhs = []

    row_i = 0

    # Rotational equations: Nx*Ny equations. These equations are always necessary to have the correct magnetic field
    for i in range(Nx):
        for j in range(Ny):
            row += [row_i]*4 #Equation number row_i(+1)
            col += [index(0,i,j),index(1,i+1,j),index(0,i,j+1),index(1,i,j)] #Unknowns
            data += [1,1,-1,-1] #Coefficients
            rhs += [fluxes[j,i]] #Right-hand side
            
            row_i += 1

    # The rest of equations fix the gauge
    # Divergence equations (not at the edges): Coulomb gauge: (Nx-1)*(Ny-1) equations
    for i in range(1,Nx):
        for j in range(1,Ny):
            row += [row_i]*4 #Equation number row_i(+1)
            col += [index(0,i,j), index(1,i,j), index(0,i-1,j), index(1,i,j-1)] #Unknowns
            data += [1,1,-1,-1] #Coefficients
            rhs += [0] #Right-hand side
            
            row_i += 1
            
    #Fix the value of A at the edges
    total_flux = np.sum(fluxes)
    if gauge == "Landau":
        for i in range(Nx): #bottom edge: Nx equations
            row += [row_i]
            col += [index(0,i,0)]
            data += [1]
            rhs += [total_flux/Nx/2]
            row_i += 1
    
        for j in range(Ny): #left edge: Ny equations
            row += [row_i]
            col+= [index(1,0,j)]
            data += [1]
            rhs += [0]
            row_i += 1
    
        for i in range(Nx-1): #top edge: Nx-1 equations
            row += [row_i]
            col+= [index(0,i,Ny)]
            data += [-1]
            rhs += [total_flux/Nx/2]
            row_i += 1
    
        for j in range(Ny): #right edge: Ny equations
            row += [row_i]
            col += [index(1,Nx,j)]
            data += [1]
            rhs += [0]
            row_i += 1
            
    elif gauge == "C4":
        for i in range(Nx): #bottom edge: Nx equations
            row += [row_i]
            col += [index(0,i,0)]
            data += [1]
            rhs += [total_flux/(Nx+Ny)/2]
            row_i += 1
    
        for j in range(Ny): #left edge: Ny equations
            row += [row_i]
            col+= [index(1,0,j)]
            data += [-1]
            rhs += [total_flux/(Nx+Ny)/2]
            row_i += 1
    
        for i in range(Nx-1): #top edge: Nx-1 equations
            row += [row_i]
            col+= [index(0,i,Ny)]
            data += [-1]
            rhs += [total_flux/(Nx+Ny)/2]
            row_i += 1
    
        for j in range(Ny): #right edge: Ny equations
            row += [row_i]
            col += [index(1,Nx,j)]
            data += [1]
            rhs += [total_flux/(Nx+Ny)/2]
            row_i += 1
    else:
        raise ValueError("Choose a valid gauge: 'Landau' or 'C4'")
        
    equations = csr_matrix((data, (row, col)), shape=(2*Nx*Ny+Nx+Ny, 2*Nx*Ny+Nx+Ny))
    vector_potential = sla.spsolve(equations, rhs)
    a_in = vector_potential[:2*Nx*Ny].reshape(Ny,Nx,2)
    a_top = vector_potential[2*Nx*Ny:2*Nx*Ny+Nx].reshape(1,Nx)
    a_right = vector_potential[2*Nx*Ny+Nx:].reshape(Ny,1)

    a_e = np.concatenate([a_in[:,:,0],a_top],axis = 0)
    a_n = np.concatenate((a_in[:,:,1],a_right),axis = 1)
    
    return a_e, a_n

def plot_A(a_e,a_n):
    '''Produces a plot of the vector potential given by a_e and a_n'''
    Nx = len(a_e[0])
    Ny = len(a_n[:,0])
    
    # Define a grid of points
    x = np.linspace(0, Nx, Nx) 
    y = np.linspace(0, Ny, Ny)  
    X, Y = np.meshgrid(x, y)    # Create a meshgrid for plotting
    
    # Create the quiver+streamline plot
    fig = plt.figure(figsize = (6,6))
    ax = fig.add_subplot(111)
    ax.quiver(X, Y, a_e[:-1,:], a_n[:,:-1],np.sqrt(a_e[:-1,:]**2 + a_n[:,:-1]**2), cmap="plasma") #The right and top edges are left out of the plot
    #ax.streamplot(X, Y, a_e[:-1,:], a_n[:,:-1],color=np.sqrt(a_e[:-1,:]**2 + a_n[:,:-1]**2), cmap="plasma") #The right and top edges are left out of the plot
    ax.set_aspect('equal')
   
    ax.set_xlim([-1, Nx+1])
    ax.set_ylim([-1, Ny+1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Vector Potential')

def plot_B(fluxes):
    '''Produces a plot of the flux array'''
    fig = plt.figure(figsize = (6,6))
    ax = fig.add_subplot(111)
    ax.imshow(fluxes,vmin = np.min(fluxes),vmax = np.max(fluxes),origin ='lower')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Magnetic field')

def recover_field(a_e, a_n):
    '''Obtains fluxes from vector potential. Useful for debugging.'''
    Nx = len(a_e[0])
    Ny = len(a_n[:,0])

    recover_flux = np.zeros((Ny,Nx))
    for i in range(Nx):
        for j in range(Ny):
            recover_flux[j,i] = a_e[j,i] + a_n[j,i+1] - a_e[j+1,i] - a_n[j,i]
    return recover_flux

######################################## RECTANGLE

def generate_rectangle(Lx, Ly, plot_shape = False):
    '''Generates the set of points in the grid closest to a rectangle with sides Lx and Ly and the angle of the normal vector.
    - Lx: int
    - Ly: int
    - plot_shape: bool
    Returns
    - boundary_points: ndarray (2,2*(Lx+Ly-2))
    - normal_angles: ndarray (2*(Lx+Ly-2))'''

    x1 = Lx*np.ones(Ly-1)
    y1 = np.linspace(1,Ly, Ly-1, endpoint=False)
    angles1 = np.zeros(Ly-1)
    x2 = np.linspace(Lx-1,0, Lx-1, endpoint = False)
    y2 = Ly*np.ones(Lx-1)
    angles2 = (pi/2)*np.ones(Lx-1)
    
    x3 = np.zeros(Ly-1)
    y3 = np.linspace(Ly-1,0, Ly-1, endpoint=False)
    angles3 = pi*np.ones(Ly-1)
    x4 = np.linspace(1,Lx, Lx-1, endpoint = False)
    y4 = np.zeros(Lx-1)
    angles4 = -(pi/2)*np.ones(Lx-1)

    x = np.concatenate((x1,x2,x3,x4))
    y = np.concatenate((y1,y2,y3,y4))

    
    normal_angles = np.concatenate((angles1,angles2,angles3,angles4))
    boundary_points = np.stack((x,y))
    
    return boundary_points, normal_angles

def operators_rectangle(parameters, return_shape = False):
    '''
    Returns operators Phi, H and P for a rectangle 
    geometry boundary condition given by a magnetization that rotates parallel to the edge.
    -parameters: dict
    -return_shape = bool
    Returns
    -Phi: csc_matrix (2*Lx*Ly-2*(Lx+Ly-2)-4,2*Lx*Ly-2*(Lx+Ly-2)-4)
    -H: csc_matrix (2*Lx*Ly-2*(Lx+Ly-2)-4,2*Lx*Ly-2*(Lx+Ly-2)-4)
    -P: csc_matrix (2*Lx*Ly-2*(Lx+Ly-2)-4,2*Lx*Ly-2*(Lx+Ly-2)-4)
    -indices_to_delete: list
    '''
    #The parameters dictionary must have the following key,value pairs
    theta = parameters['theta'] #float in (-pi,pi] Boundary condition angle
    gap = parameters['mass']    #float Mass gap
    Lx = parameters['Lx']       #int Number of lattice sites in x direction
    Ly = parameters['Ly']       #int Number of lattice sites in y direction
    Nx = Lx+1
    Ny = Ly+1
    a_e = parameters['a_e']   #ndarray(Lx,Ly+1) Peierls phases to the right 
    a_n = parameters['a_n']   #ndarray(Lx+1,Ly) Peierls phases up
    #Attach zeros to the Peierls phases to fix their size.
    a_e = np.concatenate([a_e,np.zeros((Ly+1,1))],axis = 1)
    a_n = np.concatenate([a_n,np.zeros((1,Lx+1))],axis = 0)

    row_Tx = []
    col_Tx = []
    data_Tx = []
    
    row_Ty = []
    col_Ty = []
    data_Ty = []    

    for i in range(Nx*Ny):
        y = i//Nx
        x = i%Nx
                
        #Phases
        phase_e = np.exp(1j*a_e[y,x])
        phase_n = np.exp(1j*a_n[y,x])
        
        row_Tx += [i]
        col_Tx += [((x+1)%Nx) + y*Nx]
        data_Tx += [phase_e*(1-(x//(Nx-1)))] ################## Open boundaries in x direction
        
        row_Ty += [i]
        col_Ty += [x + ((y+1)%Ny)*Nx]
        data_Ty += [phase_n*(1-(y//(Ny-1)))] ################## Open boundaries in y direction
        
    # Sparse matrices corresponding to translation operators
    Tx = csc_matrix((data_Tx, (row_Tx, col_Tx)), shape = (Nx*Ny, Nx*Ny))
    Ty = csc_matrix((data_Ty, (row_Ty, col_Ty)), shape = (Nx*Ny, Nx*Ny))
    one = scipy.sparse.identity(Nx*Ny)
    
    phi_x = (Tx+one)/2
    phi_y = (Ty+one)/2
    sin_x = -(1j/2)*(Tx-Tx.H)
    sin_y = -(1j/2)*(Ty-Ty.H)
    
    hx = phi_y.H@sin_x@phi_y
    hy = phi_x.H@sin_y@phi_x
    phi = (phi_x@phi_y+phi_y@phi_x)/2
       
    mass = scipy.sparse.spdiags(gap*np.ones(Nx*Ny), 0, Nx*Ny, Nx*Ny, format = "csc")
    M = scipy.sparse.kron(csc_matrix(sigma_z), mass, format = "csc")
    
    H_0 = scipy.sparse.kron(csc_matrix(sigma_x), hx, format = "csc") + scipy.sparse.kron(csc_matrix(sigma_y), hy, format = "csc")
    Phi = scipy.sparse.kron(csc_matrix(sigma_0), phi, format = "csc")

    H = H_0 + Phi.H@M@Phi
        
    # Unitary transformation on the edges. Let us build a rotation matrix that acts on a single site.
    def spin_rotation(site, theta, phi):
        'Unitary transformation that rotates the spin site to a (theta,phi) orientation'
        rotation = np.identity(2*Nx*Ny, dtype = complex)
        
        spinup = int(site[0] + site[1]*Nx)
        spindown = int(site[0] + site[1]*Nx + Nx*Ny)
        
        rotation[spinup,spinup] = np.cos(theta/2)
        rotation[spinup,spindown] = np.sin(theta/2)
        rotation[spindown,spinup] = -np.sin(theta/2)*np.exp(1j*phi)
        rotation[spindown,spindown] = np.cos(theta/2)*np.exp(1j*phi)
        
        return csc_matrix(rotation)
        
    # Now rotate the spins on the edge
    def get_index(x,y,s):
        '''Returns the index of the orbital in x,y with spin s'''
        return int(Nx*Ny*s + Nx*y + x)

    edge_points, normal_angles = generate_rectangle(Lx, Ly)
    # The parameter that we need for the spin rotation is the projection of the boundary spin o the plane, so the normal plus pi/2.
    boundary_spin_projections = normal_angles + np.ones(len(normal_angles))*pi/2
    
    indices_to_delete = []
    for point in zip(edge_points[0], edge_points[1], boundary_spin_projections):
        
        #rotate
        rotation = spin_rotation([point[0],point[1]], theta, point[2]) 
        H = rotation.H@H@rotation
        Phi = rotation.H@Phi@rotation
        
        #book index to delete
        indices_to_delete.append(get_index(point[0],point[1],1))

    #Now we also have to delete the outer part
    amount_out = 0
    def discriminant(x,y):
        return x>0 and x<Lx and y>0 and y<Ly
        
    X,Y = np.meshgrid(np.arange(0,Nx),np.arange(0,Ny))
    for x,y in zip(X.ravel(),Y.ravel()):
        if not discriminant(x,y) and  get_index(x,y,1) not in indices_to_delete:
            indices_to_delete.append(get_index(x,y,0))
            indices_to_delete.append(get_index(x,y,1))
            amount_out += 1

            
    # Transforming the sparse matrix into dense to delete spins
    H_aux = H.toarray()
    Phi_aux = Phi.toarray()
   
    H_aux = np.delete(H_aux, indices_to_delete, axis=0)
    H_aux = np.delete(H_aux, indices_to_delete, axis=1)
    
    Phi_aux = np.delete(Phi_aux, indices_to_delete, axis=0)
    Phi_aux = np.delete(Phi_aux, indices_to_delete, axis=1)
        
    H = csc_matrix(H_aux)
    Phi = csc_matrix(Phi_aux)
    P = Phi.H@Phi

    if return_shape:
        inside_indices = np.delete(np.arange(2*Nx*Ny), indices_to_delete)
        inside_x = inside_indices%(np.ones(len(inside_indices))*Nx)
        inside_y = (inside_indices//(np.ones(len(inside_indices))*Nx))%(np.ones(len(inside_indices))*Ny)
        inside_s = inside_indices//(np.ones(len(inside_indices))*Nx*Ny)
    
        return Phi, H, P, indices_to_delete, (inside_x[Nx*Ny-amount_out:],inside_y[Nx*Ny-amount_out:]), (inside_x[:Nx*Ny-amount_out],inside_y[:Nx*Ny-amount_out])
    else:
        return Phi, H, P, indices_to_delete

def solve_eigenproblem_rectangle(parameters, energy = 1e-6, number_of_bands = 10, plot_shape = True):
    '''
    Returns spectrum and eigenstates (and deleted indices) of a rectangular system.
    - parameters: dict
    - energy: float Value of the energy around which to calculate spectrum
    - number_of_bands: int Amount of bands closest to energ to calculate.
    - plot_shape: bool
    Returns:
    - energies: ndarray (number_of_bands)
    - states_shaped: ndarray((Lx+1)*(Ly+1),number_of_bands)
    - deleted_indeices: list
    '''
    #The parameters dictionary must have the following key,value pairs
    Lx = parameters['Lx']       #int Number of lattice sites in x direction
    Ly = parameters['Ly']       #int Number of lattice sites in y direction
    Nx = Lx+1
    Ny = Ly+1
   
    if plot_shape:
        Phi, H, P, deleted_indices, spinup_shape, spindown_shape = operators_rectangle(parameters, return_shape = True)
        fig = plt.figure(figsize = (7,7))
        ax = fig.add_subplot(111)
        ax.scatter(spinup_shape[0],spinup_shape[1], s = 20)
        ax.scatter(spindown_shape[0],spindown_shape[1], s = 20,zorder=-1)
        ax.set_aspect('equal')
        fig.show()
    else:
        Phi, H, P, deleted_indices = operators_rectangle(parameters, return_shape = False)

      #Solve generalised eigenproblem
    eigenvalues, eigenvectors = sla.eigsh(H, M=P, k = number_of_bands, tol = 1e-8, sigma = energy, which = 'LM',return_eigenvectors = True)

    #Refill with zeros the deleted spins
    states = np.zeros((2*Nx*Ny,number_of_bands),dtype = complex)
    count = 0
    for index in range(2*Nx*Ny):
        if index not in deleted_indices:
            states[index] = (Phi@eigenvectors)[index-count]
        else:
            count += 1

    #Now make sure they are orthogonal
    overlaps = states.conjugate().transpose()@states
    ##The overlap can only be non-zero for degenerate states
    degenerate_indices = []
    bulk_indices = []    
    for i in range(overlaps.shape[0]):
        sorted = np.flip(np.sort(np.abs(overlaps[i])))
        if sorted[1]/sorted[0]<0.1: #This threshold (0.1) is a bit arbitrary
            bulk_indices.append(i)
        else:
            degenerate_indices.append(i)

    overlaps_deg = np.delete(overlaps, bulk_indices, axis=0)
    overlaps_deg = np.delete(overlaps_deg, bulk_indices, axis=1)
    overlaps_bulk = np.delete(overlaps, degenerate_indices, axis=0)
    overlaps_bulk = np.delete(overlaps_bulk, degenerate_indices, axis=1)

    states_deg = np.delete(states, bulk_indices, axis=1)
    states_bulk = np.delete(states, degenerate_indices, axis=1)

    evalues, orthogonal_coeff = np.linalg.eigh(overlaps_deg)
    orthogonal = np.append(states_deg@orthogonal_coeff, states_bulk , axis=1) #### These are finally the orthogonalised states
    norm = np.sqrt(np.diag(np.abs(orthogonal.conjugate().transpose()@orthogonal)))
    states = orthogonal/norm[None,:]
    
    # Rebuild state
    def spin_rotation(site, theta, phi):
        '''Returns a unitary transformation matrix that rotates the spin site to a theta,phi orientation'''
        rotation = np.identity(2*Nx*Ny, dtype = complex)
        
        spinup = int(site[0] + site[1]*Nx)
        spindown = int(site[0] + site[1]*Nx + Nx*Ny)
        
        rotation[spinup,spinup] = np.cos(theta/2)
        rotation[spinup,spindown] = np.sin(theta/2)
        rotation[spindown,spinup] = -np.sin(theta/2)*np.exp(1j*phi)
        rotation[spindown,spindown] = np.cos(theta/2)*np.exp(1j*phi)
        
        return csc_matrix(rotation)
        
    # We need to generate again the shape in order to refill the deleted sites
    edge_points, normal_angles = generate_rectangle(Lx, Ly)
    # The parameter that we need for the spin rotation is the projection of the boundary spin on the plane, so the normal plus pi/2.
    boundary_spin_projections = normal_angles + np.ones(len(normal_angles))*pi/2
    
    # Rotate back the spins on the edge
    theta = parameters['theta']
    for point in zip(edge_points[0], edge_points[1], boundary_spin_projections):
        rotation = spin_rotation([point[0],point[1]], theta, point[2]+pi)
        states = rotation@states

    ### Reshape
    states_shaped = np.reshape(states.flatten('F'), newshape = (number_of_bands,2,Ny,Nx), order = 'C')

    ### Assign again energies
    energies = np.zeros(number_of_bands)
    for i in range(number_of_bands):
        if i in degenerate_indices:
            energies[i] = 0
        else:
            energies[i] = eigenvalues[i]
    
    return energies, states_shaped, degenerate_indices

######################################## RIBBON

def generate_ribbon(Lx, Ly, plot_shape = False):
    '''Generates the set of points in the grid closest to a ribbon with sides Lx and Ly and the angle of the normal vector.
    The boundaries are only in y direction. The ribbon is infinite in x direction.
    - Lx: int
    - Ly: int
    - plot_shape: bool
    Returns
    - boundary_points: ndarray (2,2*Lx)
    - normal_angles: ndarray (2*Lx)'''

    x2 = np.linspace(Lx-1,-1, Lx, endpoint = False)
    y2 = Ly*np.ones(Lx)
    angles2 = (pi/2)*np.ones(Lx)
    
    x4 = np.linspace(0,Lx, Lx, endpoint = False)
    y4 = np.zeros(Lx)
    angles4 = -(pi/2)*np.ones(Lx)

    x = np.concatenate((x2,x4))
    y = np.concatenate((y2,y4))

    normal_angles = np.concatenate((angles2,angles4))
    boundary_points = np.stack((x,y))
    
    return boundary_points, normal_angles

def operators_ribbon(parameters, return_shape = False):
    '''
    Returns operators Phi, H and P for a ribbon in x direction
    geometry boundary conditions are independent on each side.
    -parameters: dict
    -return_shape = bool
    Returns
    -Phi: csc_matrix (2*Lx*Ly-2*(Lx+Ly-2)-4,2*Lx*Ly-2*(Lx+Ly-2)-4)
    -H: csc_matrix (2*Lx*Ly-2*(Lx+Ly-2)-4,2*Lx*Ly-2*(Lx+Ly-2)-4)
    -P: csc_matrix (2*Lx*Ly-2*(Lx+Ly-2)-4,2*Lx*Ly-2*(Lx+Ly-2)-4)
    -indices_to_delete: list
    '''
    #The parameters dictionary must have the following key,value pairs
    theta_bot = parameters['theta_bot'] #float in (-pi,pi] Boundary condition angle on the bottom
    theta_top = parameters['theta_top'] #float in (-pi,pi] Boundary condition angle on the top
    kx = parameters['kx']       # float in (-pi/pi] wavenumber in x direction
    gap = parameters['mass']    #float Mass gap
    Lx = parameters['Lx']       #int Number of lattice sites in x direction
    Ly = parameters['Ly']       #int Number of lattice sites in y direction
    mag_field = parameters['mag_field']       #int Magnetic field strength in units hbar/(ea^2)
    noise = parameters['noise']       #int Magnetic field range in units hbar/(ea^2)
    Nx = Lx #######Notice the difference here with respect to the rectangle
    Ny = Ly+1

    np.random.seed(parameters['seed'])
    fluxes = mag_field*np.ones((Ly,Lx)) + noise*(np.random.rand(Ly,Lx)-0.5*np.ones((Ly,Lx)))
    a_e, a_n = vector_potential(1,Ly,fluxes, gauge = "Landau")
    #Attach zeros to the Peierls phases to fix their size. Only to the up ones this time.
    a_n = np.concatenate([a_n,np.zeros((1,Lx+1))],axis = 0)

    row_Tx = []
    col_Tx = []
    data_Tx = []
    
    row_Ty = []
    col_Ty = []
    data_Ty = []

    
    for i in range(Nx*Ny):
        y = i//Nx
        x = i%Nx
                
        #Phases
        phase_e = np.exp(1j*a_e[y,x])*np.exp(-1j*kx*Nx*(1 if x==Nx-1 else 0)) #We added the Bloch phase for x == Nx-1
        phase_n = np.exp(1j*a_n[y,x])
        
        row_Tx += [i]
        col_Tx += [((x+1)%Nx) + y*Nx]
        data_Tx += [phase_e] ## Here we keep periodic boundary conditions
        
        row_Ty += [i]
        col_Ty += [x + ((y+1)%Ny)*Nx]
        data_Ty += [phase_n*(1-(y//(Ny-1)))] ################## Open boundaries in y direction
        
    # Sparse matrices corresponding to translation operators
    Tx = csc_matrix((data_Tx, (row_Tx, col_Tx)), shape = (Nx*Ny, Nx*Ny))
    Ty = csc_matrix((data_Ty, (row_Ty, col_Ty)), shape = (Nx*Ny, Nx*Ny))
    one = scipy.sparse.identity(Nx*Ny)
    
    phi_x = (Tx+one)/2
    phi_y = (Ty+one)/2
    sin_x = -(1j/2)*(Tx-Tx.H)
    sin_y = -(1j/2)*(Ty-Ty.H)
    
    hx = phi_y.H@sin_x@phi_y
    hy = phi_x.H@sin_y@phi_x
    phi = (phi_x@phi_y+phi_y@phi_x)/2
       
    mass = scipy.sparse.spdiags(gap*np.ones(Nx*Ny), 0, Nx*Ny, Nx*Ny, format = "csc")
    M = scipy.sparse.kron(csc_matrix(sigma_z), mass, format = "csc")
    
    H_0 = scipy.sparse.kron(csc_matrix(sigma_x), hx, format = "csc") + scipy.sparse.kron(csc_matrix(sigma_y), hy, format = "csc")
    Phi = scipy.sparse.kron(csc_matrix(sigma_0), phi, format = "csc")

    H = H_0 + Phi.H@M@Phi
        
    # Unitary transformation on the edges. Let us build a rotation matrix that acts on a single site.
    def spin_rotation(site, theta, phi):
        'Unitary transformation that rotates the spin site to a (theta,phi) orientation'
        rotation = np.identity(2*Nx*Ny, dtype = complex)
        
        spinup = int(site[0] + site[1]*Nx)
        spindown = int(site[0] + site[1]*Nx + Nx*Ny)
        
        rotation[spinup,spinup] = np.cos(theta/2)
        rotation[spinup,spindown] = np.sin(theta/2)
        rotation[spindown,spinup] = -np.sin(theta/2)*np.exp(1j*phi)
        rotation[spindown,spindown] = np.cos(theta/2)*np.exp(1j*phi)
        
        return csc_matrix(rotation)
        
    # Now rotate the spins on the edge
    def get_index(x,y,s):
        '''Returns the index of the orbital in x,y with spin s'''
        return int(Nx*Ny*s + Nx*y + x)

    edge_points, normal_angles = generate_ribbon(Lx, Ly)
    # The parameter that we need for the spin rotation is the projection of the boundary spin o the plane, so the normal plus pi/2.
    boundary_spin_projections = normal_angles + np.ones(len(normal_angles))*pi/2
    
    indices_to_delete = []
    ##### We can make theta also an array in order to assign different values to each boundary
    theta = np.concatenate((theta_top*np.ones(Lx), theta_bot*np.ones(Lx)))
    for point in zip(edge_points[0], edge_points[1], theta, boundary_spin_projections):
        
        #rotate
        rotation = spin_rotation([point[0],point[1]], point[2], point[3]) 
        H = rotation.H@H@rotation
        Phi = rotation.H@Phi@rotation
        
        #book index to delete
        indices_to_delete.append(get_index(point[0],point[1],1))

    #Now we also have to delete the outer part
    amount_out = 0
    def discriminant(x,y):
        return y>0 and y<Ly
        
    X,Y = np.meshgrid(np.arange(0,Nx),np.arange(0,Ny))
    for x,y in zip(X.ravel(),Y.ravel()):
        if not discriminant(x,y) and  get_index(x,y,1) not in indices_to_delete:
            indices_to_delete.append(get_index(x,y,0))
            indices_to_delete.append(get_index(x,y,1))
            amount_out += 1

            
    # Transforming the sparse matrix into dense to delete spins
    H_aux = H.toarray()
    Phi_aux = Phi.toarray()
   
    H_aux = np.delete(H_aux, indices_to_delete, axis=0)
    H_aux = np.delete(H_aux, indices_to_delete, axis=1)
    
    Phi_aux = np.delete(Phi_aux, indices_to_delete, axis=0)
    Phi_aux = np.delete(Phi_aux, indices_to_delete, axis=1)
        
    H = csc_matrix(H_aux)
    Phi = csc_matrix(Phi_aux)
    P = Phi.H@Phi

    if return_shape:
        inside_indices = np.delete(np.arange(2*Nx*Ny), indices_to_delete)
        inside_x = inside_indices%(np.ones(len(inside_indices))*Nx)
        inside_y = (inside_indices//(np.ones(len(inside_indices))*Nx))%(np.ones(len(inside_indices))*Ny)
        inside_s = inside_indices//(np.ones(len(inside_indices))*Nx*Ny)
    
        return Phi, H, P, indices_to_delete, (inside_x[Nx*Ny-amount_out:],inside_y[Nx*Ny-amount_out:]), (inside_x[:Nx*Ny-amount_out],inside_y[:Nx*Ny-amount_out])
    else:
        return Phi, H, P, indices_to_delete

def make_bands_x(parameters, number_of_bands = int(20), number_of_points = int(101), kmin = -pi, kmax = pi):
    '''Calculate bands in x direction for the ribbon.'''

    momenta = np.linspace(kmin,kmax, num = number_of_points)
    bands = np.zeros((number_of_points,number_of_bands))
    
    #Solve generalised eigenproblem for all kx
    for j, kx in enumerate(momenta):
        parameters['kx'] = kx
        Phi, H, P, deleted_indices = operators_ribbon(parameters)
        bands[j] = sla.eigsh(H, M=P, k = number_of_bands, tol = 1e-7, sigma = 0.0000001, which = 'LM',return_eigenvectors = False)

    return momenta, bands

def tangent_states(parameters, kpoint,number_of_bands = int(20)):
    '''Finds the an eigenstates of a tangent fermions nanoribbon
    with zigzag boundary conditions in a magnetic field.
    The units are given by a = 1, hbar = 1, e = 1, v_F = 1
    -parameters: dict
    -kpoint: float wavenumber in x direction
    -number_of_bands: number of eigenstates calculated
    Returns
    -energies: numpy array of size number_of_bands
    -states_shaped: numpy tensor of shape (number_of_bands,2,Ny,Nx)
    -degenerate_indices: list of states indices that are degenerate'''
    parameters['kx'] = kpoint
    theta_bot = parameters['theta_bot'] #float in (-pi,pi] Boundary condition angle on the bottom
    theta_top = parameters['theta_top'] #float in (-pi,pi] Boundary condition angle on the top
    Nx = parameters['Lx']
    Ny = parameters['Ly']+1
    #Solve generalised eigenproblem 
    Phi, H, P, deleted_indices = operators_ribbon(parameters)
    eigenvalues, eigenvectors = sla.eigsh(H, M=P, k = number_of_bands, tol = 1e-7, sigma = 0.0000001, which = 'LM',return_eigenvectors = True)

    #Refill with zeros the deleted spins
    states = np.zeros((2*Nx*Ny,number_of_bands),dtype = complex)
    count = 0
    for index in range(2*Nx*Ny):
        if index not in deleted_indices:
            states[index] = (Phi@eigenvectors)[index-count]
        else:
            count += 1

    #Now make sure they are orthogonal
    overlaps = states.conjugate().transpose()@states
    ##The overlap can only be non-zero for degenerate states
    degenerate_indices = []
    bulk_indices = []    
    for i in range(overlaps.shape[0]):
        sorted = np.flip(np.sort(np.abs(overlaps[i])))
        if sorted[1]/sorted[0]<0.1: #This threshold (0.1) is a bit arbitrary
            bulk_indices.append(i)
        else:
            degenerate_indices.append(i)

    overlaps_deg = np.delete(overlaps, bulk_indices, axis=0)
    overlaps_deg = np.delete(overlaps_deg, bulk_indices, axis=1)
    overlaps_bulk = np.delete(overlaps, degenerate_indices, axis=0)
    overlaps_bulk = np.delete(overlaps_bulk, degenerate_indices, axis=1)

    states_deg = np.delete(states, bulk_indices, axis=1)
    states_bulk = np.delete(states, degenerate_indices, axis=1)

    evalues, orthogonal_coeff = np.linalg.eigh(overlaps_deg)
    orthogonal = np.append(states_deg@orthogonal_coeff, states_bulk , axis=1) #### These are finally the orthogonalised states
    norm = np.sqrt(np.diag(np.abs(orthogonal.conjugate().transpose()@orthogonal)))
    states = orthogonal/norm[None,:]
    
    # Rebuild state
    def spin_rotation(site, theta, phi):
        '''Returns a unitary transformation matrix that rotates the spin site to a theta,phi orientation'''
        rotation = np.identity(2*Nx*Ny, dtype = complex)
        
        spinup = int(site[0] + site[1]*Nx)
        spindown = int(site[0] + site[1]*Nx + Nx*Ny)
        
        rotation[spinup,spinup] = np.cos(theta/2)
        rotation[spinup,spindown] = np.sin(theta/2)
        rotation[spindown,spinup] = -np.sin(theta/2)*np.exp(1j*phi)
        rotation[spindown,spindown] = np.cos(theta/2)*np.exp(1j*phi)
        
        return csc_matrix(rotation)
        
    # We need to generate again the shape in order to refill the deleted sites
    edge_points, normal_angles = generate_ribbon(Nx, Ny-1)
    # The parameter that we need for the spin rotation is the projection of the boundary spin on the plane, so the normal plus pi/2.
    boundary_spin_projections = normal_angles + np.ones(len(normal_angles))*pi/2

     # Rotate back the spins on the edge
    theta = np.concatenate((theta_top*np.ones(Nx), theta_bot*np.ones(Nx)))
    for point in zip(edge_points[0], edge_points[1], theta, boundary_spin_projections):
        #rotate
        rotation = spin_rotation([point[0],point[1]], point[2], point[3]) 
        states = rotation@states

    ### Reshape
    states_shaped = np.reshape(states.flatten('F'), newshape = (number_of_bands,2,Ny,Nx), order = 'C')

    ### Assign again energies
    energies = np.zeros(number_of_bands)
    for i in range(number_of_bands):
        if i in degenerate_indices:
            energies[i] = 0
        else:
            energies[i] = eigenvalues[i]

    return energies, states_shaped, degenerate_indices


######################################## GRAPHENE

def graphene_magnetic_ribbon(parameters):
    '''
    Returns the Hamiltonian for a ribbon 
    geometry with zigzag boundary conditions
    in a magnetic field.
    In this case the units are given by a = 1 (y = 3a), hbar = 1, e = 1, v_F = 1
    -parameters: dict
    Returns
    -H: numpy matrix of size 4*width-2+bottom_bearded+top_bearded
    '''
    #The parameters dictionary must have the following key,value pairs
    width = parameters['width']       #int Number of lattice unit cells in y direction. Each cell contains 4 sites and it is 3a long
    kx = parameters['kx']*np.sqrt(3)             # float in (-pi/pi] wavenumber in x direction. The sqrt(3) is because the x-direction lattice site is sqrt(3)a long.
                                                 # This war parameters['k_x'] has units of 1/a
    bottom_bearded = parameters['bottom_bearded'] #bool type of zigzag bc at the bottom
    top_bearded = parameters['top_bearded']       #bool type of zigzag bc at the top
    mag_field = parameters['mag_field']           #float magnetic field
    noise = parameters['noise']       #int disorder strength
    np.random.seed(parameters['seed'])
    
    peierls_factor = np.sqrt(3)/2*mag_field*3 #The 3 is because the y-direction lattice site is 3a long

    hamiltonian = np.zeros((4*width-2+bottom_bearded+top_bearded,4*width-2+bottom_bearded+top_bearded),dtype = complex)

    def index(orbital,y):
        return orbital + 4*y-1+bottom_bearded
    ### We have 4 orbitals
    for y in range(width):
        if y!=0 or bottom_bearded:
            hamiltonian[index(0,y),index(1,y)] = -1
            hamiltonian[index(1,y),index(0,y)] = -1

        random_contribution = noise*(np.random.rand(1)[0]-0.5)
        hamiltonian[index(1,y),index(2,y)] = -1*np.exp(1j*peierls_factor*(y-width/2))*np.exp(1j*random_contribution) - np.exp(-1j*kx)*np.exp(-1j*peierls_factor*(y-width/2))*np.exp(-1j*random_contribution)
        hamiltonian[index(2,y),index(1,y)] = -1*np.exp(-1j*peierls_factor*(y-width/2))*np.exp(-1j*random_contribution) - np.exp(1j*kx)*np.exp(1j*peierls_factor*(y-width/2))*np.exp(1j*random_contribution)

        if y!=width-1 or top_bearded:
            hamiltonian[index(2,y),index(3,y)] = -1
            hamiltonian[index(3,y),index(2,y)] = -1

        if y!=width-1:
            random_contribution = noise*(np.random.rand(1)[0]-0.5)
            hamiltonian[index(3,y),index(0,y+1)] = -1*np.exp(-1j*peierls_factor*(y+0.5-width/2))*np.exp(-1j*random_contribution) - np.exp(1j*kx)*np.exp(1j*peierls_factor*(y+0.5-width/2))*np.exp(1j*random_contribution)
            hamiltonian[index(0,y+1),index(3,y)] = -1*np.exp(1j*peierls_factor*(y+0.5-width/2))*np.exp(1j*random_contribution) - np.exp(-1j*kx)*np.exp(-1j*peierls_factor*(y+0.5-width/2))*np.exp(-1j*random_contribution)

    return hamiltonian*2/3 #Adjusting units so that the fermi velocity is equal to 1
    
def graphene_bands_ribbon(parameters,npoints):
    '''
    Finds the spectrum of a graphene nanoribbon
    with zigzag boundary conditions
    in a uniform magnetic field.
    In this case the units are given by a = 1 (y = 3a), hbar = 1, e = 1, v_F = 1
    -parameters: dict
    -npoints: int number of points calculated
    Returns
    -momenta: numpy array of size npoints
    -bands: numpy matrix of size 4*width-2+bottom_bearded+top_bearded, npoints
    '''
    bands = []
    momenta = np.linspace(-pi/np.sqrt(3),pi/np.sqrt(3),npoints)
    for kx in momenta:
        parameters['kx'] = kx
        hamiltonian = graphene_magnetic_ribbon(parameters)
        spectrum = np.linalg.eigvalsh(hamiltonian)
        bands.append(spectrum)
        #bands.append(spectrum[np.argsort(np.abs(spectrum))]) #Sorted by distance to E=0
    bands = np.array(bands)

    return momenta, bands

def graphene_states(parameters,kpoint):
    '''Finds the an eigenstate of a graphene nanoribbon
    with zigzag boundary conditions
    in a uniform magnetic field.
    In this case the units are given by a = 1 (y = 3a), hbar = 1, e = 1, v_F = 1
    -parameters: dict
    -kpoint: float wavenumber in x direction
    Returns
    -eigenvalues: numpy array of size 4*width-2+bottom_bearded+top_bearded
    -eigenvalues: numpy tensor of size 4,width,4*width-2+bottom_bearded+top_bearded'''
    parameters['kx'] = kpoint
    hamiltonian = graphene_magnetic_ribbon(parameters)
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
    order = np.argsort(np.abs(eigenvalues))  #Sorted by distance to E=0
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:,order]
    
    norms = np.sum(np.abs(eigenvectors)**2, axis = 0)
    for i in range(eigenvectors.shape[1]):
        eigenvectors[:,i] = eigenvectors[:,i]/norms[i]/np.sqrt(3) #The np.sqrt(3) is because the y direction lattice constant is 3a
    
    # If not bearded, we refill
    if not parameters['bottom_bearded']:
        eigenvectors = np.roll(np.concatenate((eigenvectors,np.zeros((1,4*parameters['width']-2+parameters['bottom_bearded']+parameters['top_bearded']))), axis = 0),1,axis = 0)
    if not parameters['top_bearded']:
        eigenvectors = np.concatenate((eigenvectors,np.zeros((1,4*parameters['width']-2+parameters['bottom_bearded']+parameters['top_bearded']))), axis = 0)
    
    # Reshape
    psi0 = eigenvectors[::4]
    psi1 = eigenvectors[1::4]
    psi2 = eigenvectors[2::4]
    psi3 = eigenvectors[3::4]
    eigenvectors = np.stack((psi0,psi1,psi2,psi3))
    
    order = np.argsort(np.abs(eigenvalues))  #Sorted by distance to E=0
    

    return eigenvalues, eigenvectors

def graphene_square(parameters):
    '''
    Returns the Hamiltonian for a square
    geometry in a magnetic field.
    In this case the units are given by a = 1 (y = 3a), hbar = 1, e = 1, v_F = 1
    -parameters: dict
    Returns
    -H: numpy matrix of size 4*width-2+bottom_bearded+top_bearded
    '''
    #The parameters dictionary must have the following key,value pairs
    length = parameters['length']     #int Number of lattice unit cells in x direction. Each cell contains 4 sites and it is sqrt(3)a long in x
    width = parameters['width']       #int Number of lattice unit cells in y direction. Each cell contains 4 sites and it is 3a long in y
    kx = parameters['kx']*np.sqrt(3)             # float in (-pi/pi] wavenumber in x direction. The sqrt(3) is because the x-direction lattice site is sqrt(3)a long.
                                                 # This war parameters['k_x'] has units of 1/a
    bottom_bearded = parameters['bottom_bearded'] #bool type of zigzag bc at the bottom
    top_bearded = parameters['top_bearded']       #bool type of zigzag bc at the top
    mag_field = parameters['mag_field']           #float magnetic field
    noise = parameters['noise']       #int disorder strength

    #Let us build the noise array
    np.random.seed(parameters['seed']) #Set the seed before building the matrix
    fluxes = noise*(np.random.rand(2*width-2,length)-0.5)*3*np.sqrt(3)/2 #Factor to make flux from magnetic field
    # The next two lines "integrate the fluxes" to obtain the Peierls phases. There is gauge fredom, and we have chosen to fix the gauge
    # by imposing that the only hoppings that can vary randomly are 1->2 and 3->0 (in the same x). It is easy to derive that the next two lines
    # do the job.
    random_contribution = np.cumsum(fluxes, axis=0) * ((-1)**np.arange(fluxes.shape[0]))[:, None]
    random_contribution = np.vstack([np.zeros((1, random_contribution.shape[1])), random_contribution])
    
    peierls_factor = np.sqrt(3)/2*mag_field*3 #The 3 is because the y-direction lattice site is 3a long

    hamiltonian = np.zeros(((4*width-2+bottom_bearded+top_bearded)*length,(4*width-2+bottom_bearded+top_bearded)*length),dtype = complex)
    

    def index(orbital,y,x):
        return orbital + 4*y-1+bottom_bearded + (width*4-2+bottom_bearded+top_bearded)*x
        
    ### We have 4 orbitals
    for x in range(length):
        for y in range(width):
            if y!=0 or bottom_bearded:
                hamiltonian[index(0,y,x),index(1,y,x)] += -1
                hamiltonian[index(1,y,x),index(0,y,x)] += -1
            
            hamiltonian[index(1,y,x),index(2,y,x)] += -np.exp(1j*peierls_factor*(y-width/2))*np.exp(1j*random_contribution[2*y,x])
            hamiltonian[index(2,y,x),index(1,y,x)] += -np.exp(-1j*peierls_factor*(y-width/2))*np.exp(-1j*random_contribution[2*y,x])
            hamiltonian[index(1,y,x),index(2,y,(x-1)%length)] += -np.exp(-1j*peierls_factor*(y-width/2))*np.exp(-1j*kx*length*(1 if x==0 else 0))
            hamiltonian[index(2,y,(x-1)%length),index(1,y,x)] += -np.exp(1j*peierls_factor*(y-width/2))*np.exp(1j*kx*length*(1 if x==0 else 0))
            
    
            if y!=width-1 or top_bearded:
                hamiltonian[index(2,y,x),index(3,y,x)] += -1
                hamiltonian[index(3,y,x),index(2,y,x)] += -1
    
            if y!=width-1:
                hamiltonian[index(3,y,x),index(0,y+1,x)] += -np.exp(-1j*peierls_factor*(y+0.5-width/2))*np.exp(1j*random_contribution[2*y+1,x])
                hamiltonian[index(0,y+1,x),index(3,y,x)] += -np.exp(1j*peierls_factor*(y+0.5-width/2))*np.exp(-1j*random_contribution[2*y+1,x])
                hamiltonian[index(3,y,x),index(0,y+1,(x+1)%length)] += -np.exp(1j*peierls_factor*(y+0.5-width/2))*np.exp(1j*kx*length*(1 if x==length-1 else 0))
                hamiltonian[index(0,y+1,(x+1)%length),index(3,y,x)] += -np.exp(-1j*peierls_factor*(y+0.5-width/2))*np.exp(-1j*kx*length*(1 if x==length-1 else 0))

    return hamiltonian*2/3 #Adjusting units so that the fermi velocity is equal to 1

def graphene_spectrum(parameters):
    '''
    Finds the spectrum of a graphene square
    with zigzag boundary conditions in x direction
    and open in y direction
    in a uniform magnetic field.
    The units are given by a = 1 (y = 3a), hbar = 1, e = 1, v_F = 1
    -parameters: dict
    Returns
    -spectrum: numpy array of size (4*width-2+bottom_bearded+top_bearded)*length
    '''
    parameters['kx'] = 0
    hamiltonian = graphene_square(parameters)
    spectrum = np.linalg.eigvalsh(hamiltonian)

    return spectrum

