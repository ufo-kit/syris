Bodies
======

Bodies are used to model physical objects like samples, optical elements like
gratings, etc. A :class:`.MovableBody` can be moved by using an instance of
:class:`syris.geometry.Trajectory`. It is also possible to move it by
manipulating its transformation matrix directly. :class:`.CompositeBody` can
contain multiple bodies in order to model complex motion patterns, e.g. a
robotic arm.


Base
----

.. automodule:: syris.bodies.base
    :members:


Simple Bodies
-------------

.. automodule:: syris.bodies.simple
    :members:


Isosurfaces
-----------

.. automodule:: syris.bodies.isosurfaces
    :members:


Meshes
------

.. automodule:: syris.bodies.mesh
    :members:


Mesh animation readers
----------------------

``MeshAnimationReader`` accepts a directory of OBJ files or one temporal VTKHDF
``vtkPolyData`` file. It keeps the VTKHDF reader open and loads one frame at a
time, including sequences with changing topology. ``time_values`` and
``frame_time`` use the file's time convention without assuming seconds.
Unsupported composite or volume datasets raise ``ValueError``.

For mesh construction, attach the coordinate unit explicitly:

.. code-block:: python

    import quantities as q
    from syris.bodies.mesh import Mesh
    from syris.bodies.meshreader import MeshAnimationReader
    from syris.geometry import Trajectory

    reader = MeshAnimationReader("sample.vtkhdf")
    trajectory = Trajectory([(0, 0, 0)] * q.um)
    for frame in reader:
        mesh = Mesh(
            frame.vertices.T * q.um,
            trajectory,
            center=None,
            bounds=frame.bounds.copy() * q.um,
        )
        # Project this frame with the configured SYRIS backend.

Initialize SYRIS before creating bodies. Iteration reuses the reader object;
``list(reader)`` therefore does not collect independent frames. Copy its arrays
when retaining frames. No mesh repair or coordinate rounding is performed beyond
the requested dtype conversion. Float32 is the default, independently of
``syris.init``; use ``dtype=np.float64`` to retain Float64 coordinates in VTKHDF.
Generated normals follow VTK's normal filter precision.

.. automodule:: syris.bodies.meshreader
    :members: MeshAnimationReader, WavefrontAnimationReader
