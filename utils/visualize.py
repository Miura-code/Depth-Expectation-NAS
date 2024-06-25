# Copyright (c) Malong LLC
# All rights reserved.
#
# Contact: github@malongtech.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

import os
import sys
import genotypes.genotypes as gt
from graphviz import Digraph
from PIL import Image
import glob

from utils.data_prefetcher import data_prefetcher


def plot(genotype, file_path, caption=None):
    """ make DAG plot and save to file_path as .png """
    edge_attr = {
        'fontsize': '20',
        'fontname': 'times'
    }
    node_attr = {
        'style': 'filled',
        'shape': 'rect',
        'align': 'center',
        'fontsize': '20',
        'height': '0.5',
        'width': '0.5',
        'penwidth': '2',
        'fontname': 'times'
    }
    g = Digraph(
        format='png',
        edge_attr=edge_attr,
        node_attr=node_attr,
        engine='dot')
    g.body.extend(['rankdir=LR'])

    # input nodes
    g.node("c_{k-2}", fillcolor='darkseagreen2')
    g.node("c_{k-1}", fillcolor='darkseagreen2')

    # intermediate nodes
    n_nodes = len(genotype)
    for i in range(n_nodes):
        g.node(str(i), fillcolor='lightblue')

    for i, edges in enumerate(genotype):
        for op, j in edges:
            if j == 0:
                u = "c_{k-2}"
            elif j == 1:
                u = "c_{k-1}"
            else:
                u = str(j - 2)

            v = str(i)
            g.edge(u, v, label=op, fillcolor="gray")

    # output node
    g.node("c_{k}", fillcolor='palegoldenrod')
    for i in range(n_nodes):
        g.edge(str(i), "c_{k}", fillcolor="gray")

    # add image caption
    if caption:
        g.attr(label=caption, overlap='false', fontsize='20', fontname='times')

    g.render(file_path, view=False)


def plot2(genotype, file_path, caption=None, concat=None):
    """ make DAG plot and save to file_path as .png """
    edge_attr = {
        'fontsize': '20',
        'fontname': 'times'
    }
    node_attr = {
        'style': 'filled',
        'shape': 'polygon',
        'sides': '5',
        'align': 'center',
        'fontsize': '20',
        'height': '0.5',
        'width': '0.5',
        'penwidth': '2',
        'fontname': 'times'
    }
    g = Digraph(
        format='png',
        edge_attr=edge_attr,
        node_attr=node_attr,
        engine='dot')
    g.body.extend(['rankdir=LR'])

    # input nodes
    g.node("c_{k-2}", fillcolor='darkgoldenrod1')
    g.node("c_{k-1}", fillcolor='darkgoldenrod1')

    # intermediate nodes
    n_nodes = len(genotype)
    for i in range(n_nodes):
        g.node(str(i), fillcolor='indianRed1')

    for i, edges in enumerate(genotype): # 6
        for op, j in edges:
            if j == 0:
                u = "c_{k-2}"
            elif j == 1:
                u = "c_{k-1}"
            else:
                u = str(j - 2)

            v = str(i)
            g.edge(u, v, label=op, fillcolor="gray")

    # output node
    g.node("c_{k}", fillcolor='honeydew2')

    if concat is None:
        for i in range(n_nodes - 2, n_nodes):
            g.edge(str(i), "c_{k}", fillcolor="gray")
    else:
        for i in concat:
            g.edge(str(i - 2), "c_{k}", fillcolor="gray")

    # add image caption
    if caption:
        g.attr(label=caption, overlap='false', fontsize='20', fontname='times')

    g.render(file_path, view=False)


def png2gif(dir_path:str, file_name="DAG_Histtory", size=(1000, 130), pattern="*"):
    files = sorted(glob.glob(dir_path + '/' + pattern + '.png'))
    images = []
    for file in files:
        img = Image.open(file)
        img = img.resize(size)
        images.append(img)
    images[0].save(os.path.join(dir_path, file_name)+'.gif' , save_all = True , append_images = images[1:] , duration = 400 , loop = 1)

def showModelOnTensorboard(tb_writer, model, dataloader):
    """
        write model visualization on tensorboad
    """
    prefetcher = data_prefetcher(dataloader)
    images, _ = prefetcher.next()
    tb_writer.add_graph(model, images)

