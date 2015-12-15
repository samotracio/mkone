# -*- coding: utf-8 -*-
"""
MKONE (Mock Light Cone Utilities)
---------------------------------------------------

**Generate mock light cones with filamentary strucure or Gaussian blobs added**

Note the structure does not intend to be realistic and the mock catalogs can
be useful for teaching or testing correlation functions but not to do real 
science (except for routine ``uniform_sky()``)

Main Routines
-------------
:func:`mcone_filam`:
    Generate a mock light cone of random 3D points emulating filamentary structure
:func:`mcone_gaussblobs`:
    Generate a mock light cone of random 3D points with added gaussian blobs or "clusters"
:func:`filament_box`:
    Generate random 3D points emulating filamentary structure over a box
:func:`fill_cone`:
    Fill the volume of an observation light cone with small cubes of 3D points
:func:`uniform_sky`:
    Generate uniform random points (ra,dec) in a given area of sky

Auxiliary Routines
------------------
:func:`ra_dec_to_xyz`:
    Convert (ra,dec) to cartesian (x,y,z) coordinates
:func:`rdz2xyz`:
    Convert (ra,dec,reds) to cartesian (x,y,z) coordiantes for a given cosmology
:func:`randomize`:
    Randomize a set of (x,y,z) points in a box by mirroring and 90 deg rotations
:func:`ranshell3d`:
    Generate a spherical shell of random points
:func:`raz2xy`:
    Convert (ra,r) to polar coordinates for 2D cone plot
:func:`uniform_sphere`:
    Draw uniform points inside a sphere of given radius
:func:`scatter3d`:
    Basic 3D scatter plot
:func:`randomize`:
    Randomize a set of (x,y,z) points in a box by mirroring and 90 deg rotations
:func:`send`:
    Send data with SAMP to VO apps (uses ``sampc`` library)

Example of use
--------------
::

    # Create a 60x60 deg light cone with z=[0.01,0.15], filled with filaments
    import mkone as mk
    ralim  = [0,60]
    declim = [0,60]
    zlim   = [0.01,0.15]
    kone = mk.mcone_filam(ralim,declim,zlim,npts=80000,nvoids=2000,nstep=100,b=150.)
    # Send to Topcat (start Topcat first)
    mk.send(kone,'kone')

::

    # Create a 40x20 deg light cone with z=[0.01,0.15], filled with gaussian "clusters"
    import mkone as mk
    ralim   = [10,50]
    declim  = [20,40]
    zlim    = [0.01,0.15]
    cradlim = [0.5,20]
    kone = mk.mcone_gaussblobs(ralim,declim,zlim,n=20000,ufrac=0.7,ncen=500,
                               cradlim=cradlim,oformat='array')
    # Send to Topcat (start Topcat first)
    mk.send(kone,'kone',cols=['ra','dec','z','comd','px','py','pz'])

Dependencies
------------
1. astropy : to provide table output and cosmological functions
2. matplotlib,mplot3d : to provide basic graphics
3. scipy.spatial : to provide cKDTree for fast searching nearest voids
4. sampc : to exchange data over SAMP with some VO apps (e.g. Topcat)

To Do
-----
- [ ] Add seed random as input parameter where needed

Credits
-------
E. Donoso (this package)
Thanks to S. Woods and T. Sousbie (filament algorithm) and M. Bernyk (cone filling)

"""

# DEFINE IMPORTS  -------------------------------------------------------------
import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM
from astropy.coordinates import cartesian_to_spherical
from scipy.spatial import cKDTree
from mpl_toolkits.mplot3d import Axes3D
import sampc


# =============================================================================
# AUXILIARY FUNCTIONS
# =============================================================================
def ra_dec_to_xyz(ra, dec):
    ''' Convert (ra,dec) to cartesian (x,y,z) coordinates '''
    sin_ra = np.sin(ra * np.pi / 180.)
    cos_ra = np.cos(ra * np.pi / 180.)
    sin_dec = np.sin(np.pi / 2 - dec * np.pi / 180.)
    cos_dec = np.cos(np.pi / 2 - dec * np.pi / 180.)
    return (cos_ra * sin_dec, sin_ra * sin_dec, cos_dec)

def rdz2xyz(ra,dec,reds,cosmo):
    ''' Convert (ra,dec,reds) to cartesian (x,y,z) coordiantes for a given cosmology '''
    ra = np.asarray(ra)
    dec = np.asarray(dec)
    reds = np.asarray(reds)
    
    sin_ra = np.sin(ra*np.pi/180.)
    cos_ra = np.cos(ra*np.pi/180.)
    sin_dec = np.sin(np.pi/2 - dec*np.pi/180)
    cos_dec = np.cos(np.pi/2 - dec*np.pi/180)
    
    r = cosmo.comoving_distance(reds).value
    x = r * cos_ra * sin_dec
    y = r * sin_ra * sin_dec
    z = r * cos_dec
    return (x,y,z)

def ranshell3d(xc,yc,zc,r1,r2,n=500):
    ''' 
    Generate n random points in a spherical shell of thickness r2-r1, centered
    on (xc,yc,zc)
    '''
    th = np.random.random(n)*(2*np.pi)
    z0 = np.random.random(n)*2. + (-1.)
    x0 = np.sqrt(1-z0**2)*np.cos(th)
    y0 = np.sqrt(1-z0**2)*np.sin(th)
    t = np.random.random(n)*(r2**3 - r1**3) + r1**3
    r = t**(1./3.)
    x = x0*r + xc
    y = y0*r + yc
    z = z0*r + zc
    return (x,y,z)

def raz2xy(ra,r):
    ''' Convert (ra,r) to polar coordinates for 2D cone plot'''
    theta = ra * np.pi / 180  # radians
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return (x,y)

def uniform_sphere(n,R):
    ''' Draw uniform n points inside a sphere of radius R'''
    phi = np.random.random(n)*2*np.pi
    costheta = np.random.random(n)*2. - 1.0
    h = np.random.random(n)
    
    theta = np.arccos(costheta)
    r = R * (h**(1./3.))
    
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    
    m = np.asarray([x,y,z])
    m = m.T
    return m
    
def scatter3d(xyz,y=None,z=None,s=1.0,c='k'):
    ''' 
    Basic 3D scatter plot. Can pass a single xyz array of shape (n,3) or 
    3 different vectors for x,y,z 
    '''
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    if xyz.shape[1]==3:
        x,y,z = xyz[:,0],xyz[:,1],xyz[:,2]
    else :
        x,y,z = xyz,y,z
    ax.scatter(x,y,z,s=s,c=c)

def send(dat,fname,cols=None,disc=True):
    ''' Send a 2D array or astropy table to topcat via SAMP '''
    from sampc import Client
    c = Client()
    c.send(dat,fname,cols=cols,disc=disc)

def randomize(xyz,b):
    '''
    Randomize a set of (x,y,z) points defined over box of size b, by random 
    mirroring axes and by randomly rotating coordinates by 90 deg.
    
    This function is useful when a simulated box of points needs to be stacked
    multiple times. By randomizing the box, we preserve the inner strucures but
    avoid creating artificial patterns.
    
    Parameters
    ----------
    xyz : array
        Array of 3D points of shape (n,3)
    b : float
        Box size

    Returns
    ----------
    ranxyz : array
        Array of randomized 3D points
    '''
    # Mirror along a randomly chosen axis
    x,y,z = xyz[:,0],xyz[:,1],xyz[:,2]
    rn = np.random.random_integers(1,8)
    if rn==1 :
        x = x
    else :
        if rn==2 :
            x = b-x
        else :
            if rn==3 :
                y = b-y
            else :
                if rn==4 :
                    z = b-z
                else :
                    if rn==5 :
                        x = b-x
                        y = b-y
                    else :
                        if rn==6 :
                            y = b-y
                            z = b-z
                        else :
                            if rn==7 :
                                x = b-x
                                z = b-z
                            else :
                                if rn==8 :
                                    x = b-x
                                    y = b-y
    
    # Rotate by 90 deg around a randomly chosen direction
    rn = np.random.random_integers(1,6)
    if rn==1 :
        ranxyz = np.asarray(zip(x,y,z))
    else :
        if rn==2 :
            ranxyz = np.asarray(zip(z,x,y))
        else :
            if rn==3 :
                ranxyz = np.asarray(zip(y,z,x))
            else :
                if rn==4 :
                    ranxyz = np.asarray(zip(x,z,y))
                else :
                    if rn==5 :
                        ranxyz = np.asarray(zip(y,x,z))
                    else :
                        if rn==6 :
                            ranxyz = np.asarray(zip(z,y,x))
    return ranxyz



# =============================================================================
# MAIN FUNCTIONS
# =============================================================================
def uniform_sky(ralim, declim, n=1):
    ''' Generate n uniform random points (ra,dec) in a given piece of sky'''
    zlim = np.sin(np.pi * np.asarray(declim) / 180.)
    z = zlim[0] + (zlim[1] - zlim[0]) * np.random.random(n)
    dec = (180. / np.pi) * np.arcsin(z)
    ra = ralim[0] + (ralim[1] - ralim[0]) * np.random.random(n)
    return ra,dec


def mcone_gaussblobs(ralim,declim,zlim,n=20000,ufrac=0.7,ncen=500,cradlim=None,
                     rand_elong=False,fix_nmemb=False,random_state=None,
                     doplot=False,colorize=False,cosmo=None,oformat='table'):
    '''
    Generate a mock light cone of random 3D points with added gaussian blobs
    or "clusters"
    
    These clusters are distributed randomly over the cone and can have either 
    fixed or random sizes within a given interval. They can alse be randomly
    elongated along xyz axes. The nr. of memebers of each cluster can be either
    fixed or proportional to its extension (really?)
    
    The fraction of uniform (non-clustered) points (``ufrac``) can be specified, 
    leaving (1-ufrac)*n points that are distributed among ``ncen`` clusters
            
    Parameters
    ----------
    ralim : list (ra0, ra1)
        RA limits [deg]
    declim : list (dec0, dec1)
        DEC limits [deg]
    zlim : list (z0, z1)
        Redshift limits
    npts : integer
        Total number of points in the cone. Default 20000
    ufrac : float
        Fraction of uniformly distributed points (i.e. non-clustered objects)
    ncen : integer
        Number of clusters
    cradlim : float or list of form [rmin,rmax]
        Size of clusters in Mpc. In reality it corresponds to 2*sigma of the 
        gaussians. If float, all clusters have the same size. If a list, 
        sizes are chosen randomly between the given limits. If None, defaults 
        to [0.5,25]
    rand_elong : bool
        If True, clusters are randomly deformed along xyz axes by a factor of 3
    fix_nmemb : bool
        If True, all clusters have the same nr of members
    random_state : integer
        Seed for random number generator
    doplot : bool
        If True, generate a plane polar and a ra-dec plot
    docolorize : bool
        If True, points for centers and cluster members are colored differently
    cosmo : astropy cosmology object
        If None, defaults to a flat LambdaCDM with H_0=100 and Om=0.3
    oformat : string
        Select output format as
        * 'table' : astropy table. Default
        * 'array' : numpy array
        
    Notes
    -----
    The exact nr of uniform (non-clustered) points is added at the end to 
    complete ``n`` total points
    
    Returns
    ----------
    kone : astropy.table / array of shape (nobj_in_cone,7)
        Table or array of 7 columns [ra,dec,redshift,comdis(redshift),x,y,z]
    '''
    if random_state is not None:  np.random.seed(seed=random_state)
    if cradlim is None : cradlim = [0.5,25.]
    if cosmo is None:
        cosmo = FlatLambdaCDM(H0=100, Om0=0.3)

    nunif    = np.int(ufrac*n)   # Desided nr of uniform sources
    nnotunif = n-nunif           # Desired nr of sources in clusters

    # Randomly pick ncen centers for the clusters -----------------------------
    Cra,Cdec = uniform_sky(ralim,declim,n=ncen)
    Cred     = (zlim[1]-zlim[0])*np.random.random(ncen) + zlim[0]
    Cx,Cy,Cz = rdz2xyz(Cra,Cdec,Cred,cosmo)

    # Set a fixed or random radius within a the input range  ------------------
    if isinstance(cradlim,list):
        Crad = (cradlim[1]-cradlim[0])*np.random.random_sample(ncen) + cradlim[0]
    else:
        Crad = np.asarray([cradlim]*ncen)
    Crad = Crad/2.  # inout radius is 2*std, so get std for multivariate_normal()

    # Set the number of members of each cluster  ------------------------------
    if fix_nmemb:
        k = np.int(1.0*nnotunif/ncen)  #same nr for all clusters
        Cmem = [k]*ncen
    else :
        uu = np.random.randint(4,nnotunif+1,ncen-1)
        tt = np.append(uu,[4,nnotunif+1])
        tt.sort()
        Cmem = np.diff(tt)  # n_members by some proportionality to extension

    # Generate clusters   -----------------------------------------------------
    print 'Generating clusters...'
    for i in range(ncen):
        print '  |--',i,Cra[i],Cdec[i],Cred[i],Crad[i],Cmem[i]
        # Real space center for the 3D gaussian (center of cluster)
        mm = np.array([Cx[i],Cy[i],Cz[i]])
        # Real space std for the 3D gaussian (sort of size of cluster)
        if rand_elong : 
            efac = np.random.random(3)*3.
        else :
            efac = np.ones(3)
        stdx = Crad[i]*efac[0]
        stdy = Crad[i]*efac[1]
        stdz = Crad[i]*efac[2]
        cv = np.array([[stdx,0.,0.],[0.,stdy,0.],[0.,0.,stdz]])
        # Number of members of the cluster
        #nmem = np.int(1.0*nnotunif/ncen)  #same nr for all clusters
        nmem = Cmem[i]
        # Finally generate the cluster
        if Cmem[i] > 0:
            m=np.random.multivariate_normal(mm,cv,nmem)
            if i == 0 : 
                abc = m
            else:
                abc = np.concatenate((abc,m),axis=0)
    x1,y1,z1 = abc[:,0],abc[:,1],abc[:,2]
            
    # Convert clusters from (x,y,z) to (ra,dec,redshift)  ---------------------
    ttt = cartesian_to_spherical(x1,y1,z1)
    comd1,dec1,ra1 = ttt[0].value,ttt[1].value*180./np.pi,ttt[2].value*180./np.pi
    tmpz = np.linspace(0.,zlim[1]+7.,500)
    tmpd = cosmo.comoving_distance(tmpz).value
    reds1 = np.interp(comd1,tmpd,tmpz,left=-1., right=-1.)
    if (reds1<0).any(): raise Exception('Problem with redshifts!')

    # Cut clusters to desired ra,dec,redshift window   ------------------------
    idx,=np.where( (ra1>ralim[0]) & (ra1<ralim[1]) & (dec1>declim[0]) & (dec1<declim[1]) & (reds1>zlim[0]) & (reds1<zlim[1]) )
    ra1,dec1,reds1=ra1[idx],dec1[idx],reds1[idx]
    ncpop=len(ra1)  # this is the effective nr of clustered objects in the cone
    
    # Add uniform background points to complete n objects  --------------------
    nback = n-ncen-ncpop
    ra2,dec2 = uniform_sky(ralim,declim,n=nback)
    reds2    = (zlim[1]-zlim[0])*np.random.random(nback) + zlim[0]
    x2,y2,z2 = rdz2xyz(ra2,dec2,reds2,cosmo)
    #print 'n,nunif,nnotunif,ncen,ncpop,nback',n,nunif,nnotunif,ncen,ncpop,nback
    print 'Results'
    print '  |--Nr of clusters              :',ncen
    print '  |--Nr of cluster members       :',ncpop
    print '  |--Nr of non-cluster objects   :',nback
    print '  |--Total nr of objects in cone :',n

    # Join everything and do plot  --------------------------------------------
    ra   = np.concatenate([Cra,ra1,ra2])
    dec  = np.concatenate([Cdec,dec1,dec2])
    reds = np.concatenate([Cred,reds1,reds2])
    distC = cosmo.comoving_distance(Cred).value
    dist1 = cosmo.comoving_distance(reds1).value
    dist2 = cosmo.comoving_distance(reds2).value
    dist   = np.concatenate([distC,dist1,dist2])
    x = np.concatenate([Cx,x1,x2])
    y = np.concatenate([Cy,y1,y2])
    z = np.concatenate([Cz,z1,z2])

    if colorize:
        c2,c1,cc = 'k','b','r'
        spts, scen = 0.2, 20
    else:
        c2,c1,cc = 'k','k','k'
        spts, scen = 0.2, 0.2

    if doplot:
        #x,y=raz2xy(ra,z)
        #plt.scatter(x,y,s=0.4)  #c=z,cmap='jet'
        #plt.scatter(ra,dec,s=spts,c=z)
        x2,y2=raz2xy(ra2,reds2)
        x1,y1=raz2xy(ra1,reds1)
        xC,yC=raz2xy(Cra,Cred)
        plt.scatter(x2,y2,s=spts,color=c2)
        plt.scatter(x1,y1,s=spts,color=c1)
        plt.scatter(xC,yC,s=scen,color=cc)
        
        plt.figure()
        plt.scatter(ra2,dec2,s=spts,color=c2)
        plt.scatter(ra1,dec1,s=spts,color=c1)
        plt.scatter(Cra,Cdec,s=scen,color=cc)
        plt.axis('equal')

    # Choose ouput format  ----------------------------------------------------    
    kone = np.asarray(zip(ra,dec,reds,dist,x,y,z))
    if oformat == 'table' :
        cols = ['ra','dec','z','comd','px','py','pz']
        kone = Table(data=kone, names=cols)

    return kone



def filament_box(npts=80000,nvoids=2000,nstep=100,rmin=None,rep=None,b=150.):
    '''
    Generate random 3D points emulating filamentary structure over a box
    
    To create filaments, the algorithm evolves a set of random points over a 
    sphere, pushing them away in small steps from a random set of void centers.
    In each step, the coordinates are contracted a bit towards the origin. Then,
    the sphere is chopped into a cube.
    
    Optionally, the box can be stacked in 3D as desired to cover a larger
    volume (but each box is uniquely random)
            
    Parameters
    ----------
    npts : integer
        Number of points in the initial unit sphere. Default 80000
    nvoids : integer
        Number of voids in the initial unit sphere. Default 2000
    nsteps : integer
        Number of steps that points move away from void. In general, less steps
        means softer and weaker filaments. Default 100
    rmin : float
        After box is created, remove 1 member of each pair closer than rmin [Mpc]
    rep : list [nx,ny,nz]
        Generate nx*ny*nz unique random boxes and stack them. Default [1,1,1]
    b : float
        Box size in Mpc. Default 150
        
    Notes
    -----
    The box is always unit size, which is then scaled to any meaningfull size
    by ``b``. The units depend on your intepretation. A choice of ``nvoids=2000`` 
    and ``b=150`` gets a box of 150 Mpc with voids and filaments of size similar 
    to real ones. In general :
    
    * For a fixed ``b``, the larger ``nsteps`` the stronger the features
    
    * For a fixed ``b``, The larger ``nvoids`` the smaller the structure
    
    * Smaller structure means more points can be included to trace small scales
      without getting unrealistic features.

    It is hard to estimate a priori how many points will fall inside the box. 
    Just give a try and change ``npts`` accordingly.

    Returns
    ----------
    xyz : array
        Array of 3D points of shape (npts,3)
    '''
    print '-------------------------------------------------------------'
    print 'Building cube'

    if rep is None : rep = [1,1,1]
    b = 1.0*b

    # Loop over each stacked box that is desired
    ptsa = np.zeros([1,3])
    for i in range(rep[0]):
        for j in range(rep[1]):
            for k in range(rep[2]):
                # Get random points and void centers inside unit sphere
                voidpts = uniform_sphere(nvoids,1)
                pts = uniform_sphere(npts,1)
                tree = cKDTree(voidpts)
                
                # Displace points to simulate filaments
                for q in range(nstep):
                    idx = tree.query(pts,n_jobs=-1)[1]
                    nv = voidpts[idx]
                    # 0.01 == move pts 1% away from nearest void
                    # 0.9975 == move pts 0.25% towards the origin
                    npk = 0.9975*(pts + 0.01*(pts - nv))
                    pts = npk
            
                # Get unit cube inside sphere and shift origin to [0,0,0]
                # Note initial unit sphere has contracted a bit according to nsteps
                x,y,z = pts[:,0],pts[:,1],pts[:,2]
                idx,  = np.where((x>-0.5) & (x<0.5) & (y>-0.5) & (y<0.5) & (z>-0.5) & (z<0.5) )
                pts   = pts[idx] + [0.5,0.5,0.5]

                # Shift unit cube to cover the desired cubic lattice
                pts   = pts + [1.0*i,1.0*j,1.0*k]
                
                # Accumulate points of cube
                ptsa = np.vstack([ptsa,pts])

    # Scale points by scale factor
    ptsa = ptsa * [b,b,b]

    # Find pairs closer than rmin and keep only one member of each pair
    if rmin is not None:
        tree = cKDTree(ptsa)
        cp = tree.query_pairs(rmin)
        npairs = len(cp)
        if len(cp)>0:
            cp = np.asarray(list(cp))
            todel = cp[:,1]
            ptsa = np.delete(ptsa,todel,axis=0)
        print '  |-- Pairs below rmin :',npairs

    print '  |-- Total objects in cube :',len(ptsa)

    print '-------------------------------------------------------------'
    return ptsa[1:,:]  # drop dummy first element
 

def fill_cone(ralim,declim,zlim,xyz=None,b=None,repmethod='rotation',
              npts=80000,nvoids=2000,nstep=100,rmin=None,cosmo=None):
    '''
    Fill the volume of an observation light cone with small cubes of 3D points.

    The cube is stacked in 3D along the desired volume with as many cubes as 
    needed. This can generate artificial strucure so each cube can be : 

    (1) Copied as is (``repmethod='copy'``). This will likely introduce artificial patterns.
    
    (2) Randomly rotated and mirrored (``repmethod='rotation'``)
    
    (3) Generated randomly, so each box is uniquely random (``repmethod='fullrandom'``). Naturally this is slower
    
    In case (3), the routine ``filament_box()`` is used to generate boxes
    with filamentary structure

    Parameters
    ----------
    ralim : list (ra0, ra1)
        RA limits [deg]
    declim : list (dec0, dec1)
        DEC limits [deg]
    zlim : list (z0, z1)
        Redshift limits
    xyz : array of shape (:,3)
        The building block of points to be stacked. This is **not used** when
        ``repmethod=fullrandom``, as then each box is generated uniquely
    b : float
        Box size in Mpc. Default 150
    repmethod : string
        * 'copy' : the same box is repeated
        * 'rotation' : the same box is repeated but rotated and mirrored. Default
        * 'fullrandom' : generate unique random boxes each time
    npts : integer
        Number of points in the initial unit sphere. Default 80000. Used only 
        if ``repmethod='fullrandom'``
    nvoids : integer
        Number of voids in the initial unit sphere. Default 2000. Used only 
        if ``repmethod='fullrandom'``
    nsteps : integer
        Number of steps that points move away from void. In general, less steps
        means softer and weaker filaments. Default 100. Used only if 
        ``repmethod='fullrandom'``
    rmin : float
        After box is created, remove 1 member of each pair closer than rmin [Mpc]. 
        Used only if ``repmethod='fullrandom'``
    cosmo : astropy cosmology object
        If not given, defaults to a flat LambdaCDM with H_0=100 and Om=0.3
        
    Notes
    -----
    The box is always unit size, which is then scaled to any meaningfull size
    by ``b``. The units depend on your intepretation. A choice of ``nvoids=2000`` 
    and ``b=150`` gets a box of 300 Mpc with voids and filaments of size similar 
    to real ones. In general :
    
    * For a fixed ``b``, the larger ``nsteps`` the stronger the features
    
    * For a fixed ``b``, The larger ``nvoids`` the smaller the structure
    
    * Smaller structure means more points can be included to trace small scales
      without getting unrealistic features.
    
    It is hard to estimate a priori how many points will fall inside the
    observation cone. Just give a try and change ``npts`` accordingly.

    Returns
    ----------
    kone : array of shape (n,7)
        Array of 7 columns [ra,dec,redshift,comdis(redshift),x,y,z]
    '''
    print '-------------------------------------------------------------'
    print 'Filling light cone with cubes'

    if cosmo is None: cosmo = FlatLambdaCDM(H0=100, Om0=0.3)

    # Get (ra/dec/z) into units of radians and comov_Mpc  ---------------------
    torad = np.pi/180
    ra0, ra1  = ralim[0]*torad, ralim[1]*torad
    dec0,dec1 = declim[0]*torad, declim[1]*torad
    d0        = cosmo.comoving_distance(zlim[0]).value
    d1        = cosmo.comoving_distance(zlim[1]).value

    d_max = np.floor(1 + d1/b)*b  # maximum comov distance in box lengths
    xyza = np.zeros([1,3]) # accumulation array with dummy zero element

    # Loop for a 3d grid of cubes coverting the cone length  ------------------
    nb = 0
    for x0 in np.arange(-1*d_max,d_max,b):
        for y0 in np.arange(-1*d_max,d_max,b):
            for z0 in np.arange(-1*d_max,d_max,b):
                x1 = x0 + b
                y1 = y0 + b
                z1 = z0 + b
                # For each cube find corners (0 and 1) in ra/dec/dist/x/y/z space
                if (x0>=0) and (y0>=0):
                    tra0 = np.arctan2(y0,x1)
                    tra1 = np.arctan2(y1,x0)
                    if (z0>=0):
                        tdec0 = np.arctan2(z0,np.sqrt(x1**2+y1**2))
                        tdec1 = np.arctan2(z1,np.sqrt(x0**2+y0**2))
                    else:
                        tdec0 = np.arctan2(z0,np.sqrt(x0**2+y0**2))
                        tdec1 = np.arctan2(z1,np.sqrt(x1**2+y1**2))
                
                if (x0<0) and (y0>=0):
                    tra0 = np.arctan2(y1,x1)
                    tra1 = np.arctan2(y0,x0)
                    if (z0>=0):
                        tdec0 = np.arctan2(z0,np.sqrt(x0**2+y1**2))
                        tdec1 = np.arctan2(z1,np.sqrt(x1**2+y0**2))
                    else:
                        tdec0 = np.arctan2(z0,np.sqrt(x1**2+y0**2))
                        tdec1 = np.arctan2(z1,np.sqrt(x0**2+y1**2))

                if (x0<0) and (y0<0):
                    tra0 = np.arctan2(-1*y1,-1*x0) + np.pi
                    tra1 = np.arctan2(-1*y0,-1*x1) + np.pi
                    if (z0>=0):
                        tdec0 = np.arctan2(z0,np.sqrt(x0**2+y0**2))
                        tdec1 = np.arctan2(z1,np.sqrt(x1**2+y1**2))
                    else:
                        tdec0 = np.arctan2(z0,np.sqrt(x1**2+y1**2))
                        tdec1 = np.arctan2(z1,np.sqrt(x0**2+y0**2))
                        
                if (x0>=0) and (y0<0):
                    tra0 = np.arctan2(-1*y0,-1*x0) + np.pi
                    tra1 = np.arctan2(-1*y1,-1*x1) + np.pi
                    if (z0>=0):
                        tdec0 = np.arctan2(z0,np.sqrt(x1**2+y0**2))
                        tdec1 = np.arctan2(z1,np.sqrt(x0**2+y1**2))
                    else:
                        tdec0 = np.arctan2(z0,np.sqrt(x0**2+y1**2))
                        tdec1 = np.arctan2(z1,np.sqrt(x1**2+y0**2))
                
                x00 = (x0 if x0>=0 else x1)
                y00 = (y0 if y0>=0 else y1)
                z00 = (z0 if z0>=0 else z1)
                x11 = (x1 if x0>=0 else x0)
                y11 = (y1 if y0>=0 else y0)
                z11 = (z1 if z0>=0 else z0)
                
                td0 = np.sqrt(x00**2+y00**2+z00**2)
                td1 = np.sqrt(x11**2+y11**2+z11**2)
                
                # Check if cube corners fall inside the cone  -----------------
                if ((td1>d0) and (td0<d1) and (tra0<ra1) and (tra1>ra0) and (tdec0<dec1) and (tdec1>dec0)) :
                    print '  Cube',nb,'inside cone'
                    print '    |--- ','(x0,y0,z0,d0) = ',(x00,y00,z00,td0)
                    print '    |--- ','(x1,y1,z1,d1) = ',(x11,y11,z11,td1)

                    # Choose how to replicate the box
                    if repmethod == 'copy':
                        xyz = xyz
                    if repmethod == 'rotation':
                        xyz = randomize(xyz,b)
                    if repmethod == 'fullrandom':
                        xyz = filament_box(npts=npts,nvoids=nvoids,nstep=nstep,rmin=rmin,b=b)
                    
                    # Shift the cube and accumulate its xyz points  -----------
                    xyza = np.vstack([xyza, xyz + [x0,y0,z0] ])
                    nb = nb + 1

    # Since replication can reusult in new close pairs along the edges,
    # find those closer than rmin and keep only one member of these pairs
    if rmin is not None:
        tree = cKDTree(xyza)
        cp = tree.query_pairs(rmin)
        npairs = len(cp)
        if len(cp)>0:
            cp = np.asarray(list(cp))
            todel = cp[:,1]
            xyza = np.delete(xyza,todel,axis=0)
        print '    |-- Pairs below rmin (along edges of cubes):',npairs

    print '    |-- Nr of intersecting cubes :',nb
    print '    |-- Objects inside all intersecting cubes :',len(xyza)
    print '-------------------------------------------------------------'
    return xyza[1:,:] # Remove dummy zero element


   
def mcone_filam(ralim,declim,zlim,npts=80000,nvoids=2000,nstep=100,rmin=None,
                b=150.,repmethod='rotation',cosmo=None,oformat='table'):
    '''
    Generate a mock light cone of random 3D points emulating filamentary structure
    
    To create filaments, the algorithm evolves a set of random points over a 
    sphere, pushing them away in small steps from a random set of void centers.
    In each step, the coordinates are contracted a bit towards the origin. Then,
    the sphere is chopped into a cube.
    
    This cube is then stacked in 3D along the desired observation cone with 
    as many cubes as needed. This can generate artificial strucure so each cube
    can be : 

    (1) Copied as is (``repmethod='copy'``). This will likely introduce artificial patterns.
    
    (2) Randomly rotated and mirrored (``repmethod='rotation'``)
    
    (3) Generated randomly, so each box is uniquely random (``repmethod='fullrandom'``). Naturally this is slower
            
    Parameters
    ----------
    ralim : list (ra0, ra1)
        RA limits [deg]
    declim : list (dec0, dec1)
        DEC limits [deg]
    zlim : list (z0, z1)
        Redshift limits
    npts : integer
        Number of points in the initial unit sphere. Default 20000
    nvoids : integer
        Number of voids in the initial unit sphere. Default 1000
    nsteps : integer
        Number of steps that points move away from void. In general, less steps
        means softer and weaker filaments. Default 100
    rmin : float
        After box is created, remove 1 member of each pair closer than rmin [Mpc]
    b : float
        Box size in Mpc. Default 300
    repmethod : string
        Repetition method for the boxes along the cone volume
        * 'copy' : the same box is repeated
        * 'rotation' : the same box is repeated but rotated and mirrored. Default
        * 'fullrandom' : generate unique random boxes each time
    cosmo : astropy cosmology object
        If not given, defaults to a flat LambdaCDM with H_0=100 and Om=0.3
    oformat : string
        Select output format as
        * 'table' : astropy table. Default
        * 'array' : numpy array
        
    Notes
    -----
    The box is always unit size, which is then scaled to any meaningfull size
    by ``b``. The units depend on your intepretation. A choice of ``nvoids=2000`` 
    and ``b=150`` gets a box of 150 Mpc with voids and filaments of size similar 
    to real ones. In general :
    
    * For a fixed ``b``, the larger ``nsteps`` the stronger the features
    
    * For a fixed ``b``, The larger ``nvoids`` the smaller the structure
    
    * Smaller structure means more points can be included to trace small scales
      without getting unrealistic features.
    
    It is hard to estimate a priori how many points will fall inside the
    observation cone. Just give a try and change ``npts`` accordingly.

    Returns
    ----------
    kone : astropy.table / array of shape (nobj_in_cone,7)
        Table or array of 7 columns [ra,dec,redshift,comdis(redshift),x,y,z]
    '''
    if cosmo is None: cosmo = FlatLambdaCDM(H0=100, Om0=0.3)

    # Get (ra/dec/z) into units of radians and comov_Mpc  ---------------------
    #torad = np.pi/180
    #ra0, ra1  = ralim[0]*torad, ralim[1]*torad
    #dec0,dec1 = declim[0]*torad, declim[1]*torad
    d0        = cosmo.comoving_distance(zlim[0]).value
    d1        = cosmo.comoving_distance(zlim[1]).value
        
    # Build unit cube with filaments, scaled by b  ----------------------------
    xyz = filament_box(npts=npts,nvoids=nvoids,nstep=nstep,rmin=rmin,b=b)
  
    # Fill observation cone with cubes  ---------------------------------------
    xyza = fill_cone(ralim,declim,zlim,xyz=xyz,b=b,repmethod=repmethod,
                     npts=npts,nvoids=nvoids,nstep=nstep,rmin=rmin,cosmo=cosmo)

    # Prune points outside the exact observation cone  ------------------------
    x,y,z = xyza[:,0],xyza[:,1],xyza[:,2]
    ttt   = cartesian_to_spherical(x,y,z)
    comd  = ttt[0].value
    dec   = ttt[1].value*180./np.pi
    ra    = ttt[2].value*180./np.pi
    
    idx, = np.where( (ra>ralim[0]) & (ra<ralim[1]) & (dec>declim[0]) & (dec<declim[1]) & (comd>d0) & (comd<d1))
    nobj = len(idx)
    print 'Results'
    print '  |-- Prunned objects outside intersecting cubes :',len(ra)-nobj
    print '  |-- Total points inside light cone :',nobj
    print '-------------------------------------------------------------'
    
    # Transform comdis to redshift interpolating over a list  -----------------
    tmpz  = np.linspace(0.,zlim[1]+7.,500)
    tmpd  = cosmo.comoving_distance(tmpz).value
    redsh = np.interp(comd[idx],tmpd,tmpz,left=-1., right=-1.)
    if (redsh<0).any(): raise Exception('Problem with redshifts!')

    # Choose ouput format  ----------------------------------------------------    
    kone = np.asarray(zip(ra[idx],dec[idx],redsh,comd[idx],x[idx],y[idx],z[idx]))
    if oformat == 'table' :
        cols = ['ra','dec','z','comd','px','py','pz']
        kone = Table(data=kone, names=cols)

    return kone
   
        
        
###############################################################################
###############################################################################
###############################################################################
if __name__ == "__main__":

    testsuite = 'gaussblobs1'
    #testsuite = 'fil1'

#----- test suites ------------
    if testsuite == 'gaussblobs1':
        cosmo = FlatLambdaCDM(H0=100, Om0=0.3)
        ralim   = [10,50]
        declim  = [20,40]
        zlim    = [0.01,0.15]
        cradlim = [0.5,20]
        kone = mcone_gaussblobs(ralim,declim,zlim,n=20000,ufrac=0.7,ncen=500,
                                cradlim=cradlim,rand_elong=False,random_state=222,
                                doplot=False,colorize=True,cosmo=cosmo,oformat='table')

        send(kone,'kone',cols=['ra','dec','z','comd','px','py','pz'])


    if testsuite == 'fil1':
        cosmo = FlatLambdaCDM(H0=100, Om0=0.3)
        ralim  = [0,60]
        declim = [0,60]
        zlim   = [0.01,0.15]

        kone = mcone_filam(ralim,declim,zlim,npts=80000,nvoids=2000,nstep=100,rmin=None,
                           b=150.,repmethod='rotation',cosmo=cosmo,oformat='table')
        
        send(kone,'kone',cols=['ra','dec','z','comd','px','py','pz'])



                  