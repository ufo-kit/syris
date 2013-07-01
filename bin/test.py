import numpy as np
from numpy import linalg
import pyopencl as cl
import quantities as q
from syris import config as cfg
from syris.opticalelements import geometry as geom
from syris.gpu import util as gpu_util
import logging
from opticalelements.graphicalobjects import MetaBall
from opticalelements.geometry import Trajectory

LOGGER = logging.getLogger(__name__)


if __name__ == '__main__':
#     ctx = gpu_util.get_cuda_context()
#     queues = gpu_util.\
#         get_command_queues(ctx,
#                            queue_kwargs={"properties":
#                                          cl.command_queue_properties.
#                                          PROFILING_ENABLE})
#     cfg.init(queues)
    
    vec = np.array([0,1,0])*q.m
    A = linalg.inv(geom.rotate(-90*q.deg, np.array([0,0,-1])*q.m))
    B = linalg.inv(geom.translate(np.array([2,0,0])*q.m))
    C = linalg.inv(geom.rotate(90*q.deg, np.array([0,0,-1])*q.m))
    print geom.transform_vector(np.dot(A, np.dot(B, C)), vec)
    print
     
    ball = MetaBall(Trajectory(np.array([(0,0,0)])*q.m, 1*q.um), 1*q.m)
    ball.rotate(-90*q.deg, np.array([0,0,-1])*q.m)
    ball.translate(np.array([2,0,0])*q.m)
    ball.rotate(90*q.deg, np.array([0,0,-1])*q.m)
    print np.dot(linalg.inv(ball.transformation_matrix), np.array([0,1,0,1])*q.m)