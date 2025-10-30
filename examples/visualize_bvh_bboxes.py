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

"""Visualization of the BVH bounding boxes"""

import logging
import numpy as np
import quantities as q
import pyvista as pv
import vtk
import syris
import syris.geometry as geom
from syris.bodies.mesh import Mesh
from .util import get_default_parser

LOG = logging.getLogger(__name__)


def compute_depth(N, left, rope):
    """
    Computes the depth of all nodes in the BVH tree.
    """

    def IS_LEAF(node):
        return node < N

    def IS_SENTINEL(node):
        return node < 0

    total_nodes = 2 * N - 1
    if total_nodes <= 0:
        return np.array([], dtype=np.int32)

    depths = np.zeros(total_nodes, dtype=np.int32)
    root = N

    stack = [(root, 0)]

    while len(stack) > 0:
        Node, Depth = stack.pop()

        if IS_SENTINEL(Node):
            continue

        depths[Node] = Depth

        if IS_LEAF(Node):
            continue

        left_child = left[Node]
        right_child = rope[left_child]

        stack.append((left_child, Depth + 1))
        stack.append((right_child, Depth + 1))

    return depths


def compute_bvh_scalars(bbMin, bbMax, mapto="volume", ids=None, depths=None):
    """
    Helper to calculate scalar values for a *pre-sliced* set of nodes
    """
    if mapto == "id":
        if ids is None:
            raise ValueError("ids must be provided for mapto='id'")
        scalars = ids.astype(float)

    elif mapto == "volume":
        dimensions = np.maximum(0.0, bbMax - bbMin)
        volumes_raw = np.prod(dimensions, axis=1)
        scalars = volumes_raw + 1e-12

    elif mapto == "depth":
        if depths is None:
            raise ValueError("depths must be provided for mapto='depth'")
        scalars = depths.astype(float)
    else:
        raise ValueError(f"Invalid 'mapto' value: {mapto}")

    if hasattr(scalars, "get"):
        scalars = scalars.get()

    return scalars


def visualize_bvh(
    bbMin,
    bbMax,
    cmap,
    opacity_map,
    mapto="volume",
    scalar_data=None,
    log_scale=True,
    show_scalar_bar=True,
    save_path=None,
    plotter=None,
    decimation=0.0,
):
    """
    Visualizes a *pre-sliced* set of BVH boxes
    """
    _created_new_plotter = False
    if plotter is None:
        plotter = pv.Plotter()
        _created_new_plotter = True

    if bbMin is None or len(bbMin) == 0:
        return plotter

    if decimation > 0.0 and decimation < 1.0:
        n_total = len(bbMin)
        n_to_keep = int(n_total * (1.0 - decimation))

        keep_indices = np.random.choice(n_total, n_to_keep, replace=False)

        bbMin = bbMin[keep_indices]
        bbMax = bbMax[keep_indices]

        if scalar_data:
            if "ids" in scalar_data and scalar_data["ids"] is not None:
                scalar_data["ids"] = scalar_data["ids"][keep_indices]
            if "depths" in scalar_data and scalar_data["depths"] is not None:
                scalar_data["depths"] = scalar_data["depths"][keep_indices]

    elif decimation >= 1.0:
        return plotter

    scalars_per_box = compute_bvh_scalars(
        bbMin,
        bbMax,
        mapto=mapto,
        ids=scalar_data.get("ids") if scalar_data else None,
        depths=scalar_data.get("depths") if scalar_data else None,
    )

    centers = (bbMin + bbMax) / 2.0
    lengths = bbMax - bbMin

    points_polydata = pv.PolyData(centers)
    points_polydata.point_data["lengths"] = lengths

    unit_cube = pv.Cube()
    n_cells_per_glyph = unit_cube.n_cells

    glyph = vtk.vtkGlyph3D()
    glyph.SetInputData(points_polydata)
    glyph.SetSourceData(unit_cube)
    glyph.SetOrient(False)
    glyph.SetInputArrayToProcess(1, 0, 0, 0, "lengths")
    glyph.SetScaling(True)
    glyph.SetScaleModeToScaleByVectorComponents()
    glyph.Update()

    all_boxes_mesh = pv.wrap(glyph.GetOutput())

    scalars_per_cell = np.repeat(scalars_per_box, n_cells_per_glyph)
    all_boxes_mesh.cell_data["my_scalars"] = scalars_per_cell

    scalar_bar_title = f"Box {mapto.capitalize()}"
    if log_scale:
        scalar_bar_title += " (Log Scale)"

    plotter.add_mesh(
        all_boxes_mesh,
        scalars="my_scalars",
        cmap=cmap,
        log_scale=log_scale,
        scalar_bar_args={"title": scalar_bar_title}
        if show_scalar_bar
        else None,
        opacity=opacity_map,
        style="surface",
    )

    if _created_new_plotter:
        plotter.enable_anti_aliasing("msaa")
        plotter.add_axes()
        plotter.camera_position = "iso"

        if save_path:
            plotter.show(auto_close=False)
            plotter.screenshot(save_path)
            plotter.close()

    return plotter


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
        LOG.info(
            "No input file provided, loading default PyVista mesh (dragon)..."
        )

    mesh = Mesh.from_file(
        input, tr, center=args.center, unit=mesh_units, use_normals=True
    )

    mesh.build_accelerator()

    plotter = pv.Plotter()

    tree = mesh.tree
    bbMin_all_raw = tree["bbMin"][:, :3].get()
    bbMax_all_raw = tree["bbMax"][:, :3].get()
    keys_all_raw = tree["indices"].get()
    left_all = tree["left"].get()
    rope_all = tree["rope"].get()

    nb_keys = mesh.num_triangles
    total_valid_nodes = 2 * nb_keys - 1

    bbMin_all = bbMin_all_raw[0:total_valid_nodes]
    bbMax_all = bbMax_all_raw[0:total_valid_nodes]
    leaf_keys = keys_all_raw[0:nb_keys]

    all_depths = compute_depth(nb_keys, left_all, rope_all)

    use_log_scale = args.log_scale

    if args.mapto in ("id", "depth"):
        use_log_scale = False

    if args.show_nested:
        LOG.info("Visualizing all nodes (nested view)...")

        scalar_data_nested = {
            "ids": np.arange(total_valid_nodes),
            "depths": all_depths,
        }

        visualize_bvh(
            bbMin=bbMin_all,
            bbMax=bbMax_all,
            cmap=args.nested_cmap,
            opacity_map=args.opacity_map,
            plotter=plotter,
            mapto=args.mapto,
            scalar_data=scalar_data_nested,
            log_scale=use_log_scale,
            decimation=args.decimation,
        )

    else:
        if args.show_leafs:
            LOG.info(f"Visualizing {nb_keys} leaf nodes.")

            bbMin_leaf = bbMin_all[0:nb_keys]
            bbMax_leaf = bbMax_all[0:nb_keys]
            scalar_data_leaf = {
                "ids": leaf_keys,
                "depths": all_depths[0:nb_keys],
            }

            visualize_bvh(
                bbMin=bbMin_leaf,
                bbMax=bbMax_leaf,
                cmap=args.leaf_cmap,
                opacity_map=args.opacity,
                plotter=plotter,
                mapto=args.mapto,
                scalar_data=scalar_data_leaf,
                log_scale=use_log_scale,
                decimation=args.decimation,
            )

        if args.show_nodes:
            LOG.info(f"Visualizing {total_valid_nodes - nb_keys} inner nodes.")
            start_index = nb_keys
            node_indices = np.arange(start_index, total_valid_nodes)

            bbMin_node = bbMin_all[start_index:total_valid_nodes]
            bbMax_node = bbMax_all[start_index:total_valid_nodes]
            scalar_data_node = {
                "ids": node_indices,
                "depths": all_depths[start_index:total_valid_nodes],
            }

            visualize_bvh(
                bbMin=bbMin_node,
                bbMax=bbMax_node,
                cmap=args.node_cmap,
                opacity_map=args.opacity,
                plotter=plotter,
                mapto=args.mapto,
                scalar_data=scalar_data_node,
                log_scale=use_log_scale,
                decimation=args.decimation,
            )

    plotter.show()


def parse_args():
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
        "--center", type=str, default="bbox", help="Mesh centering on creation"
    )
    parser.add_argument(
        "--double-precision", action="store_true", help="Use double precision"
    )

    # --- Flags to select what to show ---
    parser.add_argument(
        "--show-leafs", action="store_true", help="Render the leaf nodes"
    )
    parser.add_argument(
        "--show-nodes", action="store_true", help="Render the inner BVH nodes"
    )
    parser.add_argument(
        "--show-nested",
        action="store_true",
        help="Show all nodes with nested transparency (overrides other flags)",
    )

    # --- Plotting style arguments ---
    parser.add_argument(
        "--mapto",
        type=str,
        default="volume",
        help="Scalar value to use for coloring (e.g., 'volume', 'id')",
    )
    parser.add_argument(
        "--log-scale",
        action="store_true",
        default=True,
        help="Use log scaling for scalars",
    )

    # --- Cmap arguments ---
    parser.add_argument(
        "--leaf-cmap",
        type=str,
        default="Greens",
        help="Colormap for leaf nodes",
    )
    parser.add_argument(
        "--node-cmap",
        type=str,
        default="Oranges",
        help="Colormap for inner nodes",
    )
    parser.add_argument(
        "--nested-cmap",
        type=str,
        default="viridis",
        help="Colormap for nested view",
    )

    # --- opacity argument names ---
    parser.add_argument(
        "--opacity",
        type=float,
        default=0.5,
        help="Flat opacity for --show-leafs or --show-nodes (e.g., 0.5)",
    )
    parser.add_argument(
        "--opacity-map",
        type=str,
        default="sigmoid",
        help="Opacity LUT for --show-nested (e.g., 'sigmoid_r', 'linear_r')",
    )

    parser.add_argument(
        "--decimation",
        type=float,
        default=0.0,
        help="Fraction of nodes to randomly remove (e.g., 0.9 = remove 90%%)",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
