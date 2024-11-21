import numpy as np
import pyvista as pv
import os
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import quantities as pq

def renderMesh (polydata):
    plotter = pv.Plotter()
    plotter.add_mesh(polydata)
    plotter.show()

def extractContiguousTriangles (polydata):
    points = polydata.GetPoints()
    cells = polydata.GetPolys()

    # Convert to numpy
    points_np = vtk_to_numpy(points.GetData())
    cells_np = vtk_to_numpy(cells.GetData())

    # The first element is the number of vertices
    # Verify the number of vertices
    cells_np = cells_np.reshape(-1, 4)
    if np.any(cells_np[:, 0] != 3):
        raise Exception("Only triangles are supported")
    cells_np = cells_np[:, 1:].flatten()
    vertices = points_np[cells_np]

    if len(vertices) != len(cells_np):
        raise Exception("Vertices and cells do not match")
    
    return vertices, cells_np


class MeshReaderDelegate(object):
    def __init__(self):
        self.dimensions = 3
        self.polydata = None
        self.bounds = None # The bounding box
        self.size = None # length of the bounding box
        self.vertices = None # Contiguous points for all triangles
        self.normals = None # Face normals
        self.triangles = None # The triangles are stored as a 1D array


class WavefrontAnimationReader(MeshReaderDelegate):
    def __init__(self, folder : str, start : int, end : int):
        super().__init__()
        self.folder = folder
        self.start = start
        self.end = end

        filenames = sorted (os.listdir(folder))
        self.filenames = [os.path.join(folder, f) for f in filenames if f.endswith('.obj')]
    
        self.bounds = None
        self.triangles = None
        self.vertices = None
        self.normals = None        

    def readFile (self, filename):
        self.reader = vtk.vtkOBJReader()
        self.reader.SetFileName(filename)
        self.reader.Update()

        polydata = self.reader.GetOutput()

        self.vertices, self.triangles = extractContiguousTriangles(polydata)
        self.bounds = polydata.GetBounds()
        self.normals = polydata.GetCellData().GetNormals()
        if self.normals is None:
            normals = vtk.vtkPolyDataNormals()
            normals.SetInputData(polydata)
            normals.ComputeCellNormalsOn()
            normals.ComputePointNormalsOff()
            normals.Update()
            self.normals = normals.GetOutput().GetCellData().GetNormals()
        
        self.bounds = np.array(self.bounds).astype(np.float32)
        self.normals = vtk_to_numpy(self.normals).reshape(-1, 3).astype(np.float32)
        self.vertices = self.vertices.astype(np.float32)
        self.triangles = self.triangles.astype(np.uint32)
        ret = (self.vertices, self.normals, self.bounds)
        return ret
    
    def getNextTimeStep (self):
        for filename in self.filenames:
            yield self.readFile(filename)

    # def getNextTimeStep (self):
    #     with concurrent.futures.ThreadPoolExecutor() as executor:
    #         futures = [executor.submit(self.readFile, filename) for filename in self.filenames]

    #         for future in concurrent.futures.as_completed(futures):
    #             try:
    #                 yield future.result()
    #             except Exception as e:
    #                 print (e)
    #                 # Exit the loop
    #                 break
    #             else:
    #                 pass

class PyvistaReader(MeshReaderDelegate):
    def __init__(self, filename : str, unit : pq.Quantity = pq.m):
        super().__init__()
        self.filename = filename

        mesh = pv.read(self.filename)
        self.polydata = mesh

        # Ensure the normals are calculated
        if mesh.cell_normals is None:
            mesh = mesh.compute_normals(cell_normals=True, point_normals=False, inplace=False)

        triangles = mesh.faces.reshape(-1, 4)[:, 1:]

        points = mesh.points

        triangle_vertices = points[triangles]

        triangle_vertices = triangle_vertices.flatten().reshape(-1, 3)

        self.vertices = np.array(triangle_vertices).astype(np.float32) * unit
        self.triangles = triangles
        self.normals = np.array(mesh.cell_normals).astype(np.float32) * unit
        self.bounds = np.array(mesh.bounds).astype(np.float32) * unit
    
    @property
    def scene (self):
        return [self.vertices, self.normals, self.bounds]
    
    def visualize (self):
        plotter = pv.Plotter(window_size=[800, 1024])
        plotter.add_mesh(self.polydata)
        plotter.show()
    

class RandomMeshReader(MeshReaderDelegate):
    def __init__(self, n : int, eps : float, lengths : np.ndarray, origin : np.ndarray):
        super().__init__()

        rng = np.random.default_rng()
        
        self.points = rng.random((n, 3)) * lengths + origin - lengths / 2
        
        # Create an empty array for vertices
        self.vertices = np.empty((n * 3, 3)).astype(np.float32)

        # Interleave u, v, and w without using a for loop
        self.vertices[0::3] = self.points
        self.vertices[1::3] = self.points + rng.random((n, 3)) * eps
        self.vertices[2::3] = self.points + rng.random((n, 3)) * eps

        self.triangles = np.arange((n * 3)).reshape(-1, 3)
        poly_faces = np.hstack((np.full((n, 1), 3), self.triangles)).flatten()
        self.polydata = pv.PolyData(self.vertices, poly_faces)

        # New random seed
        rng = np.random.default_rng()
        self.normals = np.random.rand(n, 3).astype(np.float32) - 0.5
        self.bounds = np.array([origin[0] - lengths[0] / 2, origin[0] + lengths[0] / 2, origin[1] - lengths[1] / 2, origin[1] + lengths[1] / 2, origin[2] - lengths[2] / 2, origin[2] + lengths[2] / 2]).astype(np.float32)
    
    @property
    def scene (self):
        return [self.vertices, self.normals, self.bounds]
    
    def visualize (self, normals=False, figsize=(800, 1024)):
        plotter = pv.Plotter(window_size=figsize)
        plotter.add_mesh(self.polydata)
        plotter.add_points(self.points, color='red')
        if normals:
            plotter.add_arrows(self.points, self.normals, mag=0.1)
        plotter.show()

class MeshReader():
    def __init__(self, meshReader: MeshReaderDelegate):
        self._delegate = meshReader
    
    def __getattr__(self, name):
        return getattr(self._delegate, name)
    
    