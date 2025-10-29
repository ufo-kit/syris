# Copyright (C) 2013-2023 Karlsruhe Institute of Technology
#
# This file is part of syris.
#
# This library is free software: you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation, either
# version 3 of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this library. If not, see <http://www.gnu.org/licenses/>.

"""Visualization of the BVH tree"""

import logging
import quantities as q
import syris
import syris.geometry as geom
from syris.bodies.mesh import Mesh
from .util import get_default_parser
import graphviz
import pyvista as pv

LOG = logging.getLogger(__name__)


def create_tree_visualization(nb_keys, left_all, rope_all, leaf_keys):
    """
    Creates a Graphviz Digraph of the BVH tree structure using the
    known stack-based descent algorithm.
    """
    N = nb_keys
    root = N

    def IS_LEAF(node):
        return node < N

    def IS_SENTINEL(node):
        return node < 0

    g = graphviz.Digraph("BVH_Tree", format="png")
    g.attr(nodesep="0.5", ranksep="1", overlap="false", splines="true")
    g.node_attr.update(fontname="Arial", shape="box")
    g.edge_attr.update(fontname="Arial", fontsize="10")

    stack = [root]
    visited = set()

    while len(stack) > 0:
        node = stack.pop()

        if node in visited or IS_SENTINEL(node):
            continue

        visited.add(node)

        if IS_LEAF(node):
            label = f"Leaf {node}\nKey: {leaf_keys[node]}"
            g.node(str(node), label=label, style="filled", color="#c8e6c9")  # Greenish
        else:
            label = f"Internal {node}"
            g.node(str(node), label=label, style="filled", color="#bbdefb")  # Bluish

        if IS_LEAF(node):
            continue

        left_child = left_all[node]
        right_child = rope_all[left_child]

        if not IS_SENTINEL(left_child):
            g.edge(str(node), str(left_child), label="L")
            stack.append(left_child)

        if not IS_SENTINEL(right_child):
            g.edge(str(node), str(right_child), label="R")
            stack.append(right_child)

    return g


def main():
    args = parse_args()

    mesh_units = q.Quantity(1, args.mesh_units)

    syris.init(
        loglevel=logging.INFO,
        double_precision=args.double_precision,
        compute_backend="cuda",
    )

    tr = geom.Trajectory([(0, 0, 0)] * mesh_units)

    if args.input:
        input = args.input
        LOG.info(f"Loading mesh from file: {input}")
    else:
        input = pv.examples.download_dragon()
        LOG.info("No input file provided, loading default PyVista mesh (dragon)...")

    mesh = Mesh.from_file(
        input, tr, center=args.center, unit=mesh_units, use_normals=True
    )

    mesh.build_accelerator()

    tree = mesh.tree

    keys_all_raw = tree["indices"].get()
    left_all = tree["left"].get()
    rope_all = tree["rope"].get()

    nb_keys = mesh.num_triangles
    leaf_keys = keys_all_raw[0:nb_keys]

    LOG.info("Generating BVH tree graph...")
    graph = create_tree_visualization(nb_keys, left_all, rope_all, leaf_keys)

    try:
        # Render and save the graph
        output_file_with_ext = f"{args.output_graph}.png"
        graph.render(
            args.output_graph, format="png", view=args.view_graph, cleanup=True
        )
        LOG.info(f"Graph saved to {output_file_with_ext}")
    except Exception as e:
        LOG.error(
            f"Could not render graph. Is Graphviz installed and in your system's PATH? Error: {e}"
        )
        LOG.info("You can still try to save the .dot file:")
        try:
            dot_file = graph.save(f"{args.output_graph}.dot")
            LOG.info(f"Graph source saved to {dot_file}")
        except Exception as e_dot:
            LOG.error(f"Could not even save .dot file: {e_dot}")


def parse_args():
    """REFACTORED: Parse command line arguments with original names."""
    parser = get_default_parser(__doc__)

    # --- Viz Arguments ---
    parser.add_argument("--input", type=str, help="Input .obj file")
    parser.add_argument(
        "--mesh-units",
        type=str,
        default="um",
        help="Physical units of the mesh file (e.g., 'm', 'cm', 'um')",
    )
    parser.add_argument(
        "--center", type=str, default=None, help="Mesh centering on creation"
    )
    parser.add_argument(
        "--double-precision", action="store_true", help="Use double precision"
    )
    parser.add_argument(
        "--decimation",
        type=float,
        default=0.0,
        help="Fraction of nodes to randomly remove (e.g., 0.9 = remove 90%%)",
    )

    # --- New arguments for Graphviz ---
    parser.add_argument(
        "--output-graph",
        type=str,
        default="bvh_tree",
        help="Filename for the output graph (e.g., 'bvh_tree').",
    )
    parser.add_argument(
        "--view-graph",
        action="store_true",
        help="Open the rendered graph after creation.",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
