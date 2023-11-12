# coding: utf-8
# pylint: disable=invalid-name, too-many-locals, fixme
# pylint: disable=too-many-branches, too-many-statements
# pylint: disable=dangerous-default-value
"""Visualization module"""
from __future__ import absolute_import

import re
import copy
import json

from .symbol import Symbol

def _str2tuple(string):
    """convert shape string to list, internal use only

    Parameters
    ----------
    string: str
        shape string

    Returns
    -------
    list of str to represent shape
    """
    return re.findall(r"\d+", string)

def print_summary(symbol, shape=None, line_length=120, positions=[.44, .64, .74, 1.]):
    """convert symbol for detail information

    Parameters
    ----------
    symbol: Symbol
        symbol to be visualized
    shape: dict
        dict of shapes, str->shape (tuple), given input shapes
    line_length: int
        total length of printed lines
    positions: list
        relative or absolute positions of log elements in each line
    Returns
    ------
        void
    """
    if not isinstance(symbol, Symbol):
        raise TypeError("symbol must be Symbol")
    show_shape = False
    if shape != None:
        show_shape = True
        interals = symbol.get_internals()
        _, out_shapes, _ = interals.infer_shape(**shape)
        if out_shapes == None:
            raise ValueError("Input shape is incompete")
        shape_dict = dict(zip(interals.list_outputs(), out_shapes))
    conf = json.loads(symbol.tojson())
    nodes = conf["nodes"]
    heads = set(conf["heads"][0])
    if positions[-1] <= 1:
        positions = [int(line_length * p) for p in positions]
    # header names for the different log elements
    to_display = ['Layer (type)', 'Output Shape', 'Param #', 'Previous Layer']
    def print_row(fields, positions):
        """print format row

        Parameters
        ----------
        fields: list
            information field
        positions: list
            field length ratio
        Returns
        ------
            void
        """
        line = ''
        for i in range(len(fields)):
            line += str(fields[i])
            line = line[:positions[i]]
            line += ' ' * (positions[i] - len(line))
        print(line)
    print('_' * line_length)
    print_row(to_display, positions)
    print('=' * line_length)
    def print_layer_summary(node, out_shape):
        """print layer information

        Parameters
        ----------
        node: dict
            node information
        out_shape: dict
            node shape information
        Returns
        ------
            node total parameters
        """
        op = node["op"]
        pre_node = []
        pre_filter = 0
        if op != "null":
            inputs = node["inputs"]
            for item in inputs:
                input_node = nodes[item[0]]
                input_name = input_node["name"]
                if input_node["op"] != "null" or item[0] in heads:
                    # add precede
                    pre_node.append(input_name)
                    if show_shape:
                        if input_node["op"] != "null":
                            key = input_name + "_output"
                            shape = shape_dict[key][1:]
                            pre_filter = pre_filter + int(shape[0])
                        else:
                            key = input_name
                            shape = shape_dict[key][1:]
                            pre_filter = pre_filter + int(shape[0])
        cur_param = 0
        if op == 'Convolution':
            cur_param = pre_filter \
                * int(_str2tuple(node["param"]["kernel"])[0]) \
                * int(_str2tuple(node["param"]["kernel"])[1]) \
                * int(node["param"]["num_filter"]) \
                + int(node["param"]["num_filter"])
        elif op == 'FullyConnected':
            cur_param = pre_filter * (int(node["param"]["num_hidden"]) + 1)
        elif op == 'BatchNorm':
            key = node["name"] + "_output"
            num_filter = shape_dict[key][1]
            cur_param = int(num_filter) * 2
        if not pre_node:
            first_connection = ''
        else:
            first_connection = pre_node[0]
        fields = [node['name'] + '(' + op + ')',
                  "x".join([str(x) for x in out_shape]),
                  cur_param,
                  first_connection]
        print_row(fields, positions)
        if len(pre_node) > 1:
            for i in range(1, len(pre_node)):
                fields = ['', '', '', pre_node[i]]
                print_row(fields, positions)
        return cur_param
    total_params = 0
    for i in range(len(nodes)):
        node = nodes[i]
        out_shape = []
        op = node["op"]
        if op == "null" and i > 0:
            continue
        if op != "null" or i in heads:
            if show_shape:
                if op != "null":
                    key = node["name"] + "_output"
                    out_shape = shape_dict[key][1:]
                else:
                    key = node["name"]
                    out_shape = shape_dict[key][1:]
        total_params += print_layer_summary(nodes[i], out_shape)
        if i == len(nodes) - 1:
            print('=' * line_length)
        else:
            print('_' * line_length)
    print('Total params: %s' % total_params)
    print('_' * line_length)

def plot_network(symbol, title="plot", save_format='pdf', shape=None, node_attrs={}):
    """convert symbol to dot object for visualization

    Parameters
    ----------
    title: str
        title of the dot graph
    symbol: Symbol
        symbol to be visualized
    shape: dict
        dict of shapes, str->shape (tuple), given input shapes
    node_attrs: dict
        dict of node's attributes
        for example:
            node_attrs={"shape":"oval","fixedsize":"fasle"}
            means to plot the network in "oval"
    Returns
    ------
    dot: Diagraph
        dot object of symbol
    """
    # todo add shape support
    try:
        from graphviz import Digraph
    except:
        raise ImportError("Draw network requires graphviz library")
    if not isinstance(symbol, Symbol):
        raise TypeError("symbol must be Symbol")
    draw_shape = False
    if shape != None:
        draw_shape = True
        interals = symbol.get_internals()
   