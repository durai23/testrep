import numpy as np
from numba import jit

@jit
def pair_force(x0, y0, z0, m0, x1, y1, z1, m1):
    """Calculate the force on p0 due to p1.
    
    Parameters
    ----------
    x0: float
        x-coordinate of the p0
    y0: float
        y-coordinate of the p0
    z0: float
        z-coordinate of the p0
    m0: float
        mass of the p0
    x1: float
        x-coordinate of the p1
    y1: float
        y-coordinate of the p1
    z1: float
        z-coordinate of the p1
    m1: float
        mass of the p1
    
    Returns
    -------
    f: ndarray
        force on p0 due to p1
        
    """
    
    # r2 = (x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2
    xx = (x1 - x0)
    yy = (y1 - y0)
    zz = (z1 - z0)
    r2 = xx * xx + yy * yy + zz * zz
    f = m0 * m1 * np.array([(x1 - x0), (y1 - y0), (z1 - z0)]) * r2 ** (-1.5) if r2 else np.zeros(3)
    return f

@jit
def force(x0, y0, z0, m0, x, y, z, m):
    """Calculates the force on the particle at (x0, y0, z0) due to all other particles.
    
    Parameters
    ----------
    x0: float
        x-coordinate of the particle
    y0: float
        y-coordinate of the particle
    z0: float
        z-coordinate of the particle
    m0: float
        mass of the particle
    x: ndarray
        x-coordinates of all particles
    y: ndarray
        y-coordinates of all particles
    z: ndarray
        z-coordinates of all particles
    m: ndarray
        masses of all particles.
    
    Returns
    -------
    f: ndarray
        force on particle with mass m0 at (x0, y0, z0)
    """
    f = np.zeros(3)
    for x1, y1, z1, m1 in zip(x, y, z, m):
        f += pair_force(x0, y0, z0, m0, x1, y1, z1, m1)
    return f

@jit
def calculate_all_forces(x, y, z, m):
    """Calculates the force on each particle p due to all other particles.
    
    Parameters
    ----------
    x: ndarray
        x-coordinates of all particles
    y: ndarray
        y-coordinates of all particles
    z: ndarray
        z-coordinates of all particles
    m: ndarray
        masses of all particles.
    
    Returns
    -------
    f: ndarray
        force on each particle due to all other particles.
    """
    return np.array([force(x[i], y[i], z[i], m[i], x, y, z, m) for i in range(n)])

@jit
def step(x, y, z, vx, vy, vz, m, f, dt):
    """Propagate the position and velocities.
    
    Starting from the current positions, velocities, and forces, propogate positions
    and velocities by one time step of lenght dt.
    
    .. note:: This algorithm should not be used for real simulations!
    
    Parameters
    ----------
    x: ndarray
        x-coordinates of all particles
    y: ndarray
        y-coordinates of all particles
    z: ndarray
        z-coordinates of all particles
    vx: ndarray
        x-component of the velocity of all particles
    vy: ndarray
        y-component of the velocity of all particles
    vz: ndarray
        z-component of the velocity of all particles
    m: ndarray
        masses of all particles.
    f: ndarray
        forces on particles
    dt: float
        time step
    
    Returns
    -------
    x, y, z, vx, vy, vz at t + dt
    """
    xn = x + vx * dt + 0.5 * f[0] / m * dt * dt
    yn = y + vy * dt + 0.5 * f[1] / m * dt * dt
    zn = y + vz * dt + 0.5 * f[2] / m * dt * dt
    vxn = vx + f[0] / m * dt
    vyn = vy + f[1] / m * dt
    vzn = vz + f[2] / m * dt
    return xn, yn, zn, vxn, vyn, vzn

def propagate_all_variables(x, y, z, vx, vy, vz, m, f, dt):
    for i in range(n):
        x[i], y[i], z[i], vx[i], vy[i], vz[i] = step(x[i], y[i], z[i], vx[i], vy[i], vz[i], m[i], f[i], dt)
        

if __name__ == '__main__':
    
    # 1000 particles
    n = 1000
    # time step of 0.01
    dt = 0.01

    # Initialize coordinates and velocities to random values.
    x = np.random.random(n)
    y = np.random.random(n)
    z = np.random.random(n)
    vx = np.zeros_like(x)
    vy = np.zeros_like(x)
    vz = np.zeros_like(x)
    m = np.ones_like(x)

    nsteps = 1
    for i in range(nsteps):
        f = calculate_all_forces(x, y, z, m)
        propagate_all_variables(x, y, z, vx, vy, vz, m, f, dt)