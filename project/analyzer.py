import csv
import ctypes
import re
import sys
import time
from ctypes.util import find_library

import numpy as np

from gurobipy import *

sys.path.insert(0, '../ELINA/python_interface/')

from elina_abstract0 import *
from elina_box import *
from elina_dimension import *
from elina_interval import *
from elina_lincons0 import *
from elina_linexpr0 import *
from elina_manager import *
from elina_scalar import *

libc = CDLL(find_library('c'))
cstdout = c_void_p.in_dll(libc, 'stdout')


class Layers:
    def __init__(self):
        self.layertypes = []
        self.weights = []
        self.biases = []
        self.numlayer = 0
        self.ffn_counter = 0


def parse_bias(text):
    if len(text) < 1 or text[0] != '[':
        raise Exception("expected '['")
    if text[-1] != ']':
        raise Exception("expected ']'")
    v = np.array([*map(lambda x: np.double(x.strip()), text[1:-1].split(','))])

    return v


def parse_vector(text):
    if len(text) < 1 or text[0] != '[':
        raise Exception("expected '['")
    if text[-1] != ']':
        raise Exception("expected ']'")
    v = np.array([*map(lambda x: np.double(x.strip()), text[1:-1].split(','))])

    return v.reshape((v.size, 1))


def balanced_split(text):
    i = 0
    bal = 0
    start = 0
    result = []

    while i < len(text):
        if text[i] == '[':
            bal += 1
        elif text[i] == ']':
            bal -= 1
        elif text[i] == ',' and bal == 0:
            result.append(text[start:i])
            start = i + 1
        i += 1

    if start < i:
        result.append(text[start:i])

    return result


def parse_matrix(text):
    if len(text) < 1 or text[0] != '[':
        raise Exception("expected '['")
    if text[-1] != ']':
        raise Exception("expected ']'")

    return np.array([*map(
        lambda x: parse_vector(x.strip()).flatten(), balanced_split(text[1:-1])
    )])


def parse_net(text):
    lines = [*filter(lambda x: len(x) != 0, text.split('\n'))]
    i = 0
    res = Layers()

    while i < len(lines):
        if lines[i] in ['ReLU', 'Affine']:
            W = parse_matrix(lines[i + 1])
            b = parse_bias(lines[i + 2])
            res.layertypes.append(lines[i])
            res.weights.append(W)
            res.biases.append(b)
            res.numlayer += 1
            i += 3
        else:
            raise Exception('parse error: ' + lines[i])

    return res


def parse_spec(text):
    text = text.replace("[", "")
    text = text.replace("]", "")

    with open('dummy', 'w') as my_file:
        my_file.write(text)

    data = np.genfromtxt('dummy', delimiter=',', dtype=np.double)
    low = np.copy(data[:, 0])
    high = np.copy(data[:, 1])

    return low, high


def get_perturbed_image(x, epsilon):
    image = x[1:len(x)]
    num_pixels = len(image)
    LB_N0 = image - epsilon
    UB_N0 = image + epsilon

    for i in range(num_pixels):
        if LB_N0[i] < 0:
            LB_N0[i] = 0
        if UB_N0[i] > 1:
            UB_N0[i] = 1

    return LB_N0, UB_N0


def generate_linexpr0(weights, bias, size):
    linexpr0 = elina_linexpr0_alloc(
        ElinaLinexprDiscr.ELINA_LINEXPR_DENSE, size
    )
    cst = pointer(linexpr0.contents.cst)
    elina_scalar_set_double(cst.contents.val.scalar, bias)

    for i in range(size):
        elina_linexpr0_set_coeff_scalar_double(linexpr0, i, weights[i])

    return linexpr0


def analyze(nn, LB_N0, UB_N0, label):
    nn_bounds = []

    num_pixels = len(LB_N0)
    nn.ffn_counter = 0
    numlayer = nn.numlayer

    man = elina_box_manager_alloc()
    itv = elina_interval_array_alloc(num_pixels)

    for i in range(num_pixels):
        elina_interval_set_double(itv[i], LB_N0[i], UB_N0[i])

    # construct input abstraction
    element = elina_abstract0_of_box(man, 0, num_pixels, itv)
    elina_interval_array_free(itv, num_pixels)

    for layerno in range(numlayer):
        if nn.layertypes[layerno] in ['ReLU', 'Affine']:
            layer_bounds = dict()

            weights = nn.weights[nn.ffn_counter]
            biases = nn.biases[nn.ffn_counter]

            dims = elina_abstract0_dimension(man, element)
            num_in_pixels = dims.intdim + dims.realdim
            num_out_pixels = len(weights)

            dimadd = elina_dimchange_alloc(0, num_out_pixels)

            for i in range(num_out_pixels):
                dimadd.contents.dim[i] = num_in_pixels

            elina_abstract0_add_dimensions(man, True, element, dimadd, False)
            elina_dimchange_free(dimadd)

            np.ascontiguousarray(weights, dtype=np.double)
            np.ascontiguousarray(biases, dtype=np.double)
            var = num_in_pixels

            # handle affine layer
            for i in range(num_out_pixels):
                tdim = ElinaDim(var)
                linexpr0 = generate_linexpr0(
                    weights[i], biases[i], num_in_pixels
                )
                element = elina_abstract0_assign_linexpr_array(
                    man, True, element, tdim, linexpr0, 1, None
                )
                var += 1

            dimrem = elina_dimchange_alloc(0, num_in_pixels)

            for i in range(num_in_pixels):
                dimrem.contents.dim[i] = i

            elina_abstract0_remove_dimensions(man, True, element, dimrem)
            elina_dimchange_free(dimrem)

            # add bounds after affine layer
            layer_bounds['affine'] = []
            bounds = elina_abstract0_to_box(man, element)

            for idx in range(num_out_pixels):
                layer_bounds['affine'].append(
                    (
                        bounds[idx].contents.inf.contents.val.dbl,
                        bounds[idx].contents.sup.contents.val.dbl
                    )
                )

            # handle ReLU layer
            if nn.layertypes[layerno] == 'ReLU':
                element = relu_box_layerwise(
                    man, True, element, 0, num_out_pixels
                )

            nn.ffn_counter += 1

            # add layer bounds to total bounds
            nn_bounds.append(layer_bounds)

        else:
            print(' net type not supported')

    dims = elina_abstract0_dimension(man, element)
    output_size = dims.intdim + dims.realdim

    # get bounds for each output neuron
    bounds = elina_abstract0_to_box(man, element)

    # if epsilon is zero, try to classify else verify robustness
    verified_flag = True
    predicted_label = 0

    if LB_N0[0] == UB_N0[0]:
        for i in range(output_size):
            inf = bounds[i].contents.inf.contents.val.dbl
            flag = True

            for j in range(output_size):
                if j != i:
                    sup = bounds[j].contents.sup.contents.val.dbl
                    if inf <= sup:
                        flag = False
                        break

            if flag:
                predicted_label = i
                break
    else:
        inf = bounds[label].contents.inf.contents.val.dbl
        for j in range(output_size):
            if j != label:
                sup = bounds[j].contents.sup.contents.val.dbl
                if inf <= sup:
                    predicted_label = label
                    verified_flag = False
                    break

    elina_interval_array_free(bounds, output_size)
    elina_abstract0_free(man, element)
    elina_manager_free(man)

    return predicted_label, verified_flag, nn_bounds


def refine_all_layers(nn, LB_N0, UB_N0, bounds, label, precise=False):
    """
    :type nn: class
    :param nn: contains information about neural network
    :type LB_N0: numpy.ndarray
    :param LB_N0: lower bounds for input pixels
    :type UB_N0: numpy.ndarray
    :param UB_N0: upper bounds for input pixels
    :type bounds: list of dict of list
    :param bounds: contains box bounds for every layer in the neural network
    :type label: int
    :param label: ground truth label of image
    :type precise: bool
    :param precise: whether to use box bounds for ReLUs of hidden layers or not
    :rtype: bool
    :return: whether the robustness could be verified or not
    """
    model = Model('RefineAll')
    model.setParam('OutputFlag', False)

    num_input_variables = LB_N0.size
    input_variables = [None] * num_input_variables

    for idx_var in range(num_input_variables):
        lb = LB_N0[idx_var]
        ub = UB_N0[idx_var]
        input_variables[idx_var] = model.addVar(lb=lb, ub=ub)

    model.update()

    relu_variables = input_variables

    for idx_layer in range(nn.numlayer):
        weights = nn.weights[idx_layer]
        biases = nn.biases[idx_layer]

        num_lin_expr = weights.shape[0]
        lin_expr_vars = [None] * num_lin_expr

        for idx_var in range(num_lin_expr):
            lin_expr_vars[idx_var] = LinExpr(weights[idx_var], relu_variables)
            lin_expr_vars[idx_var].addConstant(biases[idx_var])

        model.update()

        if nn.layertypes[idx_layer] == 'ReLU':
            bounds_curr_layer = bounds[idx_layer]['affine']
            relu_variables = [None] * num_lin_expr

            for idx_var in range(num_lin_expr):
                lb, ub = bounds_curr_layer[idx_var]

                if precise and 0. < ub:
                    model.setObjective(lin_expr_vars[idx_var], GRB.MINIMIZE)
                    model.optimize()
                    lb = model.ObjVal

                    model.setObjective(lin_expr_vars[idx_var], GRB.MAXIMIZE)
                    model.optimize()
                    ub = model.ObjVal

                if 0. <= lb:
                    relu_variables[idx_var] = model.addVar(lb=lb, ub=ub)
                    model.addConstr(
                        relu_variables[idx_var] ==
                        lin_expr_vars[idx_var]
                    )
                elif ub <= 0.:
                    relu_variables[idx_var] = model.addVar(lb=0., ub=0.)
                else:
                    relu_variables[idx_var] = model.addVar(lb=0.)
                    lambda_ = ub / (ub - lb)
                    mu_ = - ub * lb / (ub - lb)
                    model.addConstr(
                        relu_variables[idx_var] <=
                        lambda_ * lin_expr_vars[idx_var] + mu_
                    )
                    model.addConstr(
                        lin_expr_vars[idx_var] <= relu_variables[idx_var]
                    )

        model.update()

    if nn.layertypes[-1] == 'ReLU':
        out_variables = relu_variables
    else:
        out_variables = lin_expr_vars

    # check if the property can be verified
    num_out_vars = len(out_variables)
    verified_flag = True

    for idx_var in range(num_out_vars):
        if idx_var != label:
            model.setObjective(
                out_variables[label] - out_variables[idx_var], GRB.MINIMIZE
            )
            model.optimize()

            if model.ObjVal <= 0.:
                verified_flag = False
                break

    return verified_flag


def refine_first_n_layers(nn, LB_N0, UB_N0, bounds, num_layers, label,
                          precise=False):
    """
    :type nn: class
    :param nn: contains information about neural network
    :type LB_N0: numpy.ndarray
    :param LB_N0: lower bounds for input pixels
    :type UB_N0: numpy.ndarray
    :param UB_N0: upper bounds for input pixels
    :type bounds: list of dict of list
    :param bounds: contains box bounds for every layer in the neural network
    :type num_layers: int
    :param num_layers: how many layers should be refined
    :type label: int
    :param label: ground truth label of image
    :type precise: bool
    :param precise: whether to use box bounds for ReLUs of hidden layers or not
    :rtype: tuple of bool and list
    :return: whether the robustness could be verified or not and refined bounds
    """
    refined_bounds = bounds

    model = Model('RefineFirstN')
    model.setParam('OutputFlag', False)

    num_input_variables = LB_N0.size
    input_variables = [None] * num_input_variables

    for idx_var in range(num_input_variables):
        lb = LB_N0[idx_var]
        ub = UB_N0[idx_var]
        input_variables[idx_var] = model.addVar(lb=lb, ub=ub)

    model.update()

    relu_variables = input_variables

    # propagate epsilon-ball with linear solver
    for idx_layer in range(num_layers):
        weights = nn.weights[idx_layer]
        biases = nn.biases[idx_layer]

        num_lin_expr = weights.shape[0]
        lin_expr_vars = [None] * num_lin_expr

        for idx_var in range(num_lin_expr):
            lin_expr_vars[idx_var] = LinExpr(weights[idx_var], relu_variables)
            lin_expr_vars[idx_var].addConstant(biases[idx_var])

        model.update()

        if nn.layertypes[idx_layer] == 'ReLU':
            bounds_curr_layer = bounds[idx_layer]['affine']
            relu_variables = [None] * num_lin_expr

            for idx_var in range(num_lin_expr):
                lb, ub = bounds_curr_layer[idx_var]

                if precise and 0. < ub:
                    model.setObjective(lin_expr_vars[idx_var], GRB.MINIMIZE)
                    model.optimize()
                    lb = model.ObjVal

                    model.setObjective(lin_expr_vars[idx_var], GRB.MAXIMIZE)
                    model.optimize()
                    ub = model.ObjVal

                    refined_bounds[idx_layer]['affine'][idx_var] = (lb, ub)

                if 0. <= lb:
                    relu_variables[idx_var] = model.addVar(lb=lb, ub=ub)
                    model.addConstr(
                        relu_variables[idx_var] ==
                        lin_expr_vars[idx_var]
                    )
                elif ub <= 0.:
                    relu_variables[idx_var] = model.addVar(lb=0., ub=0.)
                else:
                    relu_variables[idx_var] = model.addVar(lb=0.)
                    lambda_ = ub / (ub - lb)
                    mu_ = - ub * lb / (ub - lb)
                    model.addConstr(
                        relu_variables[idx_var] <=
                        lambda_ * lin_expr_vars[idx_var] + mu_
                    )
                    model.addConstr(
                        lin_expr_vars[idx_var] <= relu_variables[idx_var]
                    )

        model.update()

    if nn.layertypes[-1] == 'ReLU':
        out_variables = relu_variables
    else:
        out_variables = lin_expr_vars

    # solve the linear program
    num_out_vars = len(out_variables)
    out_bounds = [None] * num_out_vars
    out_bounds_box = bounds[num_layers - 1]['affine']

    for idx_var in range(num_out_vars):
        lb, ub = out_bounds_box[idx_var]

        if 0. < ub:
            model.setObjective(out_variables[idx_var], GRB.MINIMIZE)
            model.optimize()
            lb = model.ObjVal

            model.setObjective(out_variables[idx_var], GRB.MAXIMIZE)
            model.optimize()
            ub = model.ObjVal

        else:
            lb = 0.
            ub = 0.

        out_bounds[idx_var] = (lb, ub)

    # propagate epsilon-ball with box
    man = elina_box_manager_alloc()
    itv = elina_interval_array_alloc(num_out_vars)

    for i in range(num_out_vars):
        elina_interval_set_double(itv[i], out_bounds[i][0], out_bounds[i][1])

    # construct input abstraction
    element = elina_abstract0_of_box(man, 0, num_out_vars, itv)
    elina_interval_array_free(itv, num_out_vars)

    for idx_layer in range(num_layers, nn.numlayer):
        if nn.layertypes[idx_layer] in ['ReLU', 'Affine']:
            weights = nn.weights[idx_layer]
            biases = nn.biases[idx_layer]

            dims = elina_abstract0_dimension(man, element)
            num_in_pixels = dims.intdim + dims.realdim
            num_out_pixels = len(weights)

            dimadd = elina_dimchange_alloc(0, num_out_pixels)

            for i in range(num_out_pixels):
                dimadd.contents.dim[i] = num_in_pixels

            elina_abstract0_add_dimensions(man, True, element, dimadd, False)
            elina_dimchange_free(dimadd)

            np.ascontiguousarray(weights, dtype=np.double)
            np.ascontiguousarray(biases, dtype=np.double)
            var = num_in_pixels

            # handle affine layer
            for i in range(num_out_pixels):
                tdim = ElinaDim(var)
                linexpr0 = generate_linexpr0(
                    weights[i], biases[i], num_in_pixels
                )
                element = elina_abstract0_assign_linexpr_array(
                    man, True, element, tdim, linexpr0, 1, None
                )
                var += 1

            dimrem = elina_dimchange_alloc(0, num_in_pixels)

            for i in range(num_in_pixels):
                dimrem.contents.dim[i] = i

            elina_abstract0_remove_dimensions(man, True, element, dimrem)
            elina_dimchange_free(dimrem)

            # add bounds after affine layer
            bounds = elina_abstract0_to_box(man, element)

            for idx in range(num_out_pixels):
                refined_bounds[idx_layer]['affine'][idx] = (
                    bounds[idx].contents.inf.contents.val.dbl,
                    bounds[idx].contents.sup.contents.val.dbl
                )

            # handle ReLU layer
            if nn.layertypes[idx_layer] == 'ReLU':
                element = relu_box_layerwise(
                    man, True, element, 0, num_out_pixels
                )

        else:
            print(' net type not supported')

    dims = elina_abstract0_dimension(man, element)
    output_size = dims.intdim + dims.realdim

    # get bounds for each output neuron
    bounds = elina_abstract0_to_box(man, element)

    # try to verify robustness
    verified_flag = True
    inf = bounds[label].contents.inf.contents.val.dbl

    for j in range(output_size):
        if j != label:
            sup = bounds[j].contents.sup.contents.val.dbl
            if inf <= sup:
                verified_flag = False
                break

    elina_interval_array_free(bounds, output_size)
    elina_abstract0_free(man, element)
    elina_manager_free(man)

    return verified_flag, refined_bounds


def refine_last_n_layers(nn, bounds, num_layers, label, precise=False):
    """
    :type nn: class
    :param nn: contains information about neural network
    :type bounds: list of dict of list
    :param bounds: contains box bounds for every layer in the neural network
    :type num_layers: int
    :param num_layers: how many layers should be refined
    :type label: int
    :param label: ground truth label of image
    :type precise: bool
    :param precise: whether to use box bounds for ReLUs of hidden layers or not
    :rtype: bool
    :return: whether the robustness could be verified or not
    """
    model = Model('RefineLastN')
    model.setParam('OutputFlag', False)

    idx_start_layer = nn.numlayer - num_layers
    bounds_start_layer = bounds[idx_start_layer]['affine']

    num_relu_variables = len(bounds_start_layer)
    relu_variables = [None] * num_relu_variables

    for idx_var in range(num_relu_variables):
        lb, ub = bounds_start_layer[idx_var]

        if 0. <= lb:
            relu_variables[idx_var] = model.addVar(lb=lb, ub=ub)
        elif ub <= 0.:
            relu_variables[idx_var] = model.addVar(lb=0., ub=0.)
        else:
            hidden = model.addVar(lb=lb, ub=ub)
            relu_variables[idx_var] = model.addVar(lb=0.)
            lambda_ = ub / (ub - lb)
            mu_ = - ub * lb / (ub - lb)
            model.addConstr(
                relu_variables[idx_var] <= lambda_ * hidden + mu_
            )
            model.addConstr(
                hidden <= relu_variables[idx_var]
            )

    model.update()

    for idx_curr_layer in range(idx_start_layer + 1, nn.numlayer):
        weights = nn.weights[idx_curr_layer]
        biases = nn.biases[idx_curr_layer]

        num_lin_expr = weights.shape[0]
        lin_expr_vars = [None] * num_lin_expr

        for idx_var in range(num_lin_expr):
            lin_expr_vars[idx_var] = LinExpr(weights[idx_var], relu_variables)
            lin_expr_vars[idx_var].addConstant(biases[idx_var])

        model.update()

        if nn.layertypes[idx_curr_layer] == 'ReLU':
            bounds_curr_layer = bounds[idx_curr_layer]['affine']
            relu_variables = [None] * num_lin_expr

            for idx_var in range(num_lin_expr):
                lb, ub = bounds_curr_layer[idx_var]

                if precise and 0. < ub:
                    model.setObjective(lin_expr_vars[idx_var], GRB.MINIMIZE)
                    model.optimize()
                    lb = model.ObjVal

                    model.setObjective(lin_expr_vars[idx_var], GRB.MAXIMIZE)
                    model.optimize()
                    ub = model.ObjVal

                if 0. <= lb:
                    relu_variables[idx_var] = model.addVar(lb=lb, ub=ub)
                    model.addConstr(
                        relu_variables[idx_var] ==
                        lin_expr_vars[idx_var]
                    )
                elif ub <= 0.:
                    relu_variables[idx_var] = model.addVar(lb=0., ub=0.)
                else:
                    relu_variables[idx_var] = model.addVar(lb=0.)
                    lambda_ = ub / (ub - lb)
                    mu_ = - ub * lb / (ub - lb)
                    model.addConstr(
                        relu_variables[idx_var] <=
                        lambda_ * lin_expr_vars[idx_var] + mu_
                    )
                    model.addConstr(
                        lin_expr_vars[idx_var] <= relu_variables[idx_var]
                    )

        if nn.layertypes[idx_curr_layer] == 'ReLU':
            out_variables = relu_variables
        else:
            out_variables = lin_expr_vars

        model.update()

    # check if property can be verified
    num_out_vars = len(out_variables)
    verified_flag = True

    for idx_var in range(num_out_vars):
        if idx_var != label:
            model.setObjective(
                out_variables[label] - out_variables[idx_var], GRB.MINIMIZE
            )
            model.optimize()

            if model.ObjVal <= 0.:
                verified_flag = False
                break

    return verified_flag


if __name__ == '__main__':
    from sys import argv

    if len(argv) < 3 or len(argv) > 4:
        print('usage: python3.6 ' + argv[0] + ' net.txt spec.txt [timeout]')
        exit(1)

    netname = argv[1]
    specname = argv[2]
    epsilon = float(argv[3])

    with open(netname, 'r') as netfile:
        netstring = netfile.read()
    with open(specname, 'r') as specfile:
        specstring = specfile.read()

    nn = parse_net(netstring)
    x0_low, x0_high = parse_spec(specstring)
    LB_N0, UB_N0 = get_perturbed_image(x0_low, 0)
    label = analyze(nn, LB_N0, UB_N0, 0)[0]

    start = time.time()

    if label == int(x0_low[0]):
        LB_N0, UB_N0 = get_perturbed_image(x0_low, epsilon)
        verified_flag, bounds = analyze(nn, LB_N0, UB_N0, label)[1:3]

        if verified_flag:
            print("verified")

        else:
            # choose refinement based on network architecture
            num_hidden_layers = nn.numlayer
            num_neurons_per_layer = nn.weights[0].shape[0]

            # networks: [3_10, 3_20, 3_50]
            if num_hidden_layers == 3:
                verified_flag = refine_all_layers(
                    nn, LB_N0, UB_N0, bounds, label, precise=True
                )

            # networks: [4_1024]
            elif num_hidden_layers == 4:
                verified_flag, bounds = refine_first_n_layers(
                    nn, LB_N0, UB_N0, bounds, 2, label
                )

                if not verified_flag:
                    verified_flag = refine_last_n_layers(
                        nn, bounds, 2, label, precise=True
                    )

            # networks: [6_20, 6_50, 6_100, 6_200]
            elif num_hidden_layers == 6:
                # networks: [6_20, 6_50, 6_100]
                if num_neurons_per_layer <= 100:
                    verified_flag = refine_all_layers(
                        nn, LB_N0, UB_N0, bounds, label, precise=True
                    )

                # networks: [6_200]
                else:
                    if epsilon <= 0.015:
                        verified_flag = refine_all_layers(
                            nn, LB_N0, UB_N0, bounds, label, precise=True
                        )

                    elif epsilon <= 0.0175:
                        verified_flag, bounds = refine_first_n_layers(
                            nn, LB_N0, UB_N0, bounds, 5, label, precise=True
                        )

                    else:
                        verified_flag, bounds = refine_first_n_layers(
                            nn, LB_N0, UB_N0, bounds, 4, label, precise=True
                        )

                    if not verified_flag:
                        verified_flag = refine_last_n_layers(
                            nn, bounds, 2, label, precise=True
                        )

            # networks: [9_100, 9_200]
            elif num_hidden_layers == 9:
                # networks: [9_100]
                if num_neurons_per_layer <= 100:
                    if epsilon <= 0.025:
                        verified_flag = refine_all_layers(
                            nn, LB_N0, UB_N0, bounds, label, precise=True
                        )

                    elif epsilon <= 0.0275:
                        verified_flag, bounds = refine_first_n_layers(
                            nn, LB_N0, UB_N0, bounds, 7, label, precise=True
                        )

                    elif epsilon <= 0.0325:
                        verified_flag, bounds = refine_first_n_layers(
                            nn, LB_N0, UB_N0, bounds, 6, label, precise=True
                        )

                    else:
                        verified_flag, bounds = refine_first_n_layers(
                            nn, LB_N0, UB_N0, bounds, 5, label, precise=True
                        )

                    if not verified_flag:
                        verified_flag = refine_last_n_layers(
                            nn, bounds, 2, label, precise=True
                        )

                # networks: [9_200]
                else:
                    if epsilon <= 0.0125:
                        verified_flag = refine_all_layers(
                            nn, LB_N0, UB_N0, bounds, label, precise=True
                        )

                    elif epsilon <= 0.015:
                        verified_flag, bounds = refine_first_n_layers(
                            nn, LB_N0, UB_N0, bounds, 8, label, precise=True
                        )

                    elif epsilon <= 0.0175:
                        verified_flag, bounds = refine_first_n_layers(
                            nn, LB_N0, UB_N0, bounds, 7, label, precise=True
                        )

                    elif epsilon <= 0.02:
                        verified_flag, bounds = refine_first_n_layers(
                            nn, LB_N0, UB_N0, bounds, 6, label, precise=True
                        )

                    elif epsilon <= 0.025:
                        verified_flag, bounds = refine_first_n_layers(
                            nn, LB_N0, UB_N0, bounds, 5, label, precise=True
                        )

                    else:
                        verified_flag, bounds = refine_first_n_layers(
                            nn, LB_N0, UB_N0, bounds, 4, label, precise=True
                        )

                    if not verified_flag:
                        verified_flag = refine_last_n_layers(
                            nn, bounds, 2, label, precise=True
                        )

            # unknown network architecture
            else:
                verified_flag = refine_all_layers(
                    nn, LB_N0, UB_N0, bounds, label, precise=True
                )

            if verified_flag:
                print("verified")
            else:
                # run refine_all_layers as last resort
                verified_flag = refine_all_layers(
                    nn, LB_N0, UB_N0, bounds, label, precise=True
                )

                if verified_flag:
                    print("verified")
                else:
                    print("can not be verified")

    else:
        print("image not correctly classified by the network. expected label ",
              int(x0_low[0]), " classified label: ", label)

    end = time.time()

    print("analysis time: ", (end - start), " seconds")
