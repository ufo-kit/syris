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
