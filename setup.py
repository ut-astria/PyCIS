"""
Setup function to link pycis.c, libpycis_all.so, and gsl 
"""

from distutils.core import setup, Extension 
import os
import sysconfig

#print(os.environ.get('TACC_GSL_LIB'))

d = os.getcwd()
exargs = sysconfig.get_config_var('CFLAGS').split()
exargsL = sysconfig.get_config_var('LDFLAGS').split()
exargs += ['-fopenmp','-O3',
           "-Wno-unused-variable","-Wno-unused-but-set-variable",
           "-Wno-unused-function","-Wno-sign-compare",
           "-Wno-maybe-uninitialized"]
exargsL += ['-fopenmp']
#gsllib = os.environ.get('TACC_GSL_LIB') 
#gslinc = os.environ.get('TACC_GSL_INC')
gsllib ='%s/gsl/lib'%d 
gslinc ='%s/gsl/include'%d 
module = Extension('pycis',
                    include_dirs = [gslinc, 
                                    '%s/lib'%d,
                                    'gen','two','three'],
                    libraries    = ['gsl', 'gslcblas','gomp', 'pycis_all'],
                    library_dirs = [gsllib,
                                    '%s/lib'%d,],
                    runtime_library_dirs = [gsllib,
                                    '%s/lib'%d],
                    sources = ['pycis.c'],
                    extra_compile_args = exargs,
                    extra_link_args    = exargsL)

setup(name = "pycis", ext_modules = [module])
