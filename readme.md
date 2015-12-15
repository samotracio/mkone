MKONE (Mock Light Cone Utilities)
==================================

**Generate mock light cones with filamentary strucure or Gaussian blobs added**

Note the structure does not intend to be realistic and the mock catalogs can
be useful for teaching or testing correlation functions but not to do real 
science (except for routine ``uniform_sky()``)

Main Routines
-------------
* ``mcone_filam``
    Generate a mock light cone of random 3D points emulating filamentary structure
* ``mcone_gaussblobs``
    Generate a mock light cone of random 3D points with added gaussian blobs or "clusters"
* ``filament_box``
    Generate random 3D points emulating filamentary structure over a box
* ``fill_cone``
    Fill the volume of an observation light cone with small cubes of 3D points
* ``uniform_sky``
    Generate uniform random points (ra,dec) in a given area of sky

Example of use
--------------
    # Create a 60x60 deg light cone with z=[0.01,0.15], filled with filaments
    import mkone as mk
    ralim  = [0,60]
    declim = [0,60]
    zlim   = [0.01,0.15]
    kone = mk.mcone_filam(ralim,declim,zlim,npts=80000,nvoids=2000,nstep=100,b=150.)
    # Send to Topcat (start Topcat first)
    mk.send(kone,'kone')

![kone1](./mkone1.gif?raw=true "Example cone 1")

![kone3](./mkone3.png?raw=true "Example cone 3")
    
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

![kone1](./mkone2.gif?raw=true "Example cone 2")
    
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