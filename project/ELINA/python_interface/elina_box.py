#
#
#  This source file is part of ELINA (ETH LIbrary for Numerical Analysis).
#  ELINA is Copyright © 2018 Department of Computer Science, ETH Zurich
#  This software is distributed under GNU Lesser General Public License Version 3.0.
#  For more information, see the ELINA project website at:
#  http://elina.ethz.ch
#
#  THE SOFTWARE IS PROVIDED "AS-IS" WITHOUT ANY WARRANTY OF ANY KIND, EITHER
#  EXPRESS, IMPLIED OR STATUTORY, INCLUDING BUT NOT LIMITED TO ANY WARRANTY
#  THAT THE SOFTWARE WILL CONFORM TO SPECIFICATIONS OR BE ERROR-FREE AND ANY
#  IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE,
#  TITLE, OR NON-INFRINGEMENT.  IN NO EVENT SHALL ETH ZURICH BE LIABLE FOR ANY     
#  DAMAGES, INCLUDING BUT NOT LIMITED TO DIRECT, INDIRECT,
#  SPECIAL OR CONSEQUENTIAL DAMAGES, ARISING OUT OF, RESULTING FROM, OR IN
#  ANY WAY CONNECTED WITH THIS SOFTWARE (WHETHER OR NOT BASED UPON WARRANTY,
#  CONTRACT, TORT OR OTHERWISE).
#
#


from elina_box_imports import *
from elina_manager_h import *
from elina_abstract0_h import *
import numpy as np 
from numpy.ctypeslib import ndpointer
import ctypes



# ====================================================================== #
# Basics
# ====================================================================== #

def elina_box_manager_alloc():
    """
    Allocates an ElinaManager.

    Returns
    -------
    man : ElinaManagerPtr
        Pointer to the newly allocated ElinaManager.

    """

    man = None
    try:
        elina_box_manager_alloc_c = elina_box_api.elina_box_manager_alloc
        elina_box_manager_alloc_c.restype = ElinaManagerPtr
        elina_box_manager_alloc_c.argtypes = None
        man = elina_box_manager_alloc_c()
    except:
        print('Problem with loading/calling "elina_box_manager_alloc" from "libzonotope.so"')

    return man



def relu_box_layerwise(man,destructive,elem,start_offset, num_dim):
    """
    Performs the ReLU operation
    
    Parameters
    ----------
    man : ElinaManagerPtr
        Pointer to the ElinaManager.
    destructive : c_bool
        Boolean flag.
    elem : ElinaAbstract0Ptr
        Pointer to the ElinaAbstract0 which dimensions need to be assigned.
    start_offset : ElinaDim
        The starting dimension.
    num_dim : ElinaDim
        The number of variables on which relu should be applied

    Returns
    -------
    res : ElinaAbstract0Ptr
        Pointer to the new abstract object.

    """

    res = None
    try:
        relu_box_layerwise_c = elina_box_api.relu_box_layerwise
        relu_box_layerwise_c.restype = ElinaAbstract0Ptr
        relu_box_layerwise_c.argtypes = [ElinaManagerPtr, c_bool, ElinaAbstract0Ptr, ElinaDim, ElinaDim]
        res = relu_box_layerwise_c(man,destructive,elem,start_offset, num_dim)
    except:
        print('Problem with loading/calling "relu_box_layerwise" from "libzonotope.so"')

    return res
