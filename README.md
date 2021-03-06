This program and all its subroutines were made by Gil Bohrer, Duke University, 2007.
They are introduced and explained in: Bohrer Wolosin Brady and Avissar 2007 Tellus B 59 (566-576)
This program is the intellectual property of Gil Bohrer and Roni Avissar. It is free for academic use. Do not make any commercial use in this program. Refer to the manuscript by Bohrer et al. 2007 Tellus (above) in any publication that is based on this program or that has used it to. 


Control of virtual canopy domain size, resolution and dimensions and path to output directory is in the �input parameters� section, at the top of the file.

Control of the observed canopy properties and definitions of patch types is through the file ForestCanopy_Data.m
The file Get_DBH.m include some assumptions about the shape of the allometric fit between DBH and height, based on Naidu et al 98 can j for res. This should be changed if you wish to use other allometric relationship to determine stem diameter.   

The program will generate a test canopy with 2 patch types, based on arbitrary examples of canopy strutures loosly resembling hardwood and grass.
It will save a compact form of the virtual canopy in the specified path (currently set to the local directory) and will plot some canopy properties.

The saved data files include 2-D matrices, i.e. maps, of the virtual canopy domain properties: patch (map of patch type indices), DBH (stem diameter at a defined height, [m]), Height (canopy top height [m]), and Tot_LAI (ground accumulated leaf area index [m^2 leaf / m^2 ground].
It also includes profile vectors. Each profile vector is actually a matrix. The columns represent different patch-types; the rows represent different normalized heights. The saved profiles are:  CSProfiles ([cross sectional area of stems at height z/ cross sectional area at DBH]), LAD (leaf area density [m^2 leaf between z and z+1/ m / Tot LAI] ), z (normalized heights [m/Height]). 
An example (in Fortran 90) of how to turn these map matrices and profile vectors into 3-D arrays of LAD, stem volume and stem diameter is included in the file generate_volume_array.f90. This file needs modifications to your local environment in order to run, it reads some of the canopy arrays and generates a 3-D array for cummulative (from the canopy top going down) leaf-area dencity. It should not be used "as is" but can be used a guideline and example for extrapolating the compact canopy data into volumetric arrays. 

For further assistance or questions, please contact Gil Bohrer