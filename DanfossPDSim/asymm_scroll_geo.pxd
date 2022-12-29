import cython
cimport cython

from PDSim.scroll.common_scroll_geo cimport geoVals, VdVstruct, HTAnglesClass, CVInvolutes

cdef class DanfossGeoVals(geoVals):
    """
    This is a custom class that add the geometric parameters that are 
    required for the three-arc PMP
    """
    cdef public double xa_arc3,ya_arc3,ra_arc3,t1_arc3,t2_arc3
    
cpdef double theta_d(DanfossGeoVals geo, int path) except *
cpdef int Nc(double theta, DanfossGeoVals geo, int path)
cpdef int getNc(double theta, DanfossGeoVals geo, int path)
cpdef tuple phi_s1_sa(double theta, DanfossGeoVals geo)
cpdef tuple phi_s2_sa(double theta, DanfossGeoVals geo)

cpdef CVInvolutes CVangles(double theta, DanfossGeoVals geo, int index)

cpdef HTAnglesClass HT_angles(double theta, DanfossGeoVals geo, key)
cdef _radial_leakage_angles(CVInvolutes CV_up, CVInvolutes CV_down, double *angle_min, double *angle_max)
cpdef get_radial_leakage_angles(double theta, DanfossGeoVals geo, long key1, long key2)

cpdef CVcoords(CVkey, DanfossGeoVals geo, double theta, int Ninv=*)

cpdef VdVstruct SA(double theta, DanfossGeoVals geo)
cpdef dict SA_forces(double theta, DanfossGeoVals geo)

cpdef VdVstruct DD(double theta, DanfossGeoVals geo)
cpdef dict DD_forces(double theta, DanfossGeoVals geo)

cpdef VdVstruct DDD(double theta, DanfossGeoVals geo) 
cpdef dict DDD_forces(double theta, DanfossGeoVals geo)

cpdef VdVstruct VdV(int index, double theta, DanfossGeoVals geo)
cpdef dict forces(int index, double theta, DanfossGeoVals geo, CVInvolutes angles = *)