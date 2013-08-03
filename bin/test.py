import logging
from matplotlib import pyplot as plt, cm
import numpy as np
import pyopencl as cl
from pyopencl.array import vec
import quantities as q
import syris
from syris import physics, config as cfg
from syris.gpu import util as g_util
from syris import physics, imageprocessing as ip
from syris.opticalelements import geometry as geom
from syris.opticalelements.graphicalobjects import MetaBall
from syris.opticalelements.geometry import Trajectory
from syris.devices.cameras import Camera
from metaballs import newtonraphson as nr
from metaballs.geometry import draw
from metaballs.newtonraphson import derivative
from libtiff import TIFF
import sys


LOGGER = logging.getLogger(__name__)

UNITS = q.mm
VECTOR_WIDTH = 1
SUPERSAMPLING = 1

def diff(ar):
    res = []
    for i in range(1, len(ar)):
        res.append(np.abs(ar[i] - ar[i - 1]))
        
    return res 
    
def print_array(ar):
    if len(ar) == 0:
        return
    
    res = ""
    
    for i in range(len(ar)):
        res += "{:.8f}, ".format(float(ar[i]))
    
    print res

def create_metaball_random(n, pixel_size, radius_range, coeff=1):
    x = np.random.uniform(0, n * pixel_size)
    y = np.random.uniform(0, n * pixel_size)
#     x = np.random.uniform(n / 4 * pixel_size, 3 * n / 4 * pixel_size)
#     y = np.random.uniform(n / 4 * pixel_size, 3 * n / 4 * pixel_size)
    z = np.random.uniform(radius_range[0], radius_range[1]) * \
                                                radius_range.units
    r = np.random.uniform(radius_range[0], radius_range[1]) * \
                                                radius_range.units
    
    c_points = [(x, y, z)] * q.mm
    points, length = geom.interpolate_points(c_points, pixel_size)
    trajectory = Trajectory(points, length)
    metaball = MetaBall(trajectory, r)
    metaball.move(0 * q.s)
    
    return metaball.pack(UNITS, coeff), \
        "({0}, {1}, {2}, {3}),\n".format(x, y, z.magnitude, r.magnitude)

def create_metaballs(params, coeff=1.0):
    x, y, z, r = zip(*params)
    
    objects = ""
    for i in range(len(params)):
        c_points = [(x[i], y[i], z[i])] * q.mm
        points, length = geom.interpolate_points(c_points, pixel_size)
        trajectory = Trajectory(points, length)
        metaball = MetaBall(trajectory, r[i] * q.mm)
        metaball.move(0 * q.s)
        objects += metaball.pack(UNITS, coeff)
    
    return objects

def get_vfloat_mem_host(mem, size):
    res = np.empty(size, dtype=cfg.NP_FLOAT)
    cl.enqueue_copy(cfg.QUEUE, res, mem)
    
    return res

if __name__ == '__main__':
    syris.init()
    
    pixel_size = 5e-3 / SUPERSAMPLING * q.mm
    
    prg = g_util.get_program(g_util.get_metaobjects_source())
    n = SUPERSAMPLING * 512
    thickness_mem = cl.Buffer(cfg.CTX, cl.mem_flags.READ_WRITE,
                          size=n ** 2 * VECTOR_WIDTH * cfg.CL_FLOAT)


    f = open("/home/farago/data/params.txt", "w")
    
    for i in range(1):
        objects_all = ""
        params_all = ""
        num_objects = np.random.randint(1, 100)
        min_r = SUPERSAMPLING * 5 * pixel_size.rescale(UNITS).magnitude
        radius_range = (min_r,
                        SUPERSAMPLING * 100 * 
                        pixel_size.rescale(UNITS).magnitude) * UNITS
        mid = (radius_range[0].rescale(UNITS).magnitude +
                      radius_range[1].rescale(UNITS).magnitude) / 2
        coeff = 1.0 / mid
#         eps = coeff * pixel_size.rescale(UNITS).magnitude / 4
        eps = pixel_size.rescale(UNITS).magnitude
#         eps = coeff * min_r / 10
        print "mid, coeff, eps:", mid, coeff, eps
        print i, "objects:", num_objects
        for j in range(num_objects):
            objects, params = create_metaball_random(n, pixel_size,
                                                     radius_range, coeff)
            objects_all += objects
            params_all += params
                 
        f.write("{0}:\n".format(i + 1) + params_all + "\n")




#         params = [(0.304383501401, 0.252140876393, 0.0384028416931, 0.0563961722477),
#             (0.0345938704731, 0.119592330621, 0.0537811562686, 0.0111392941656),
#             (0.316409340404, 0.499978937345, 0.0140760080432, 0.0653815647094),
#             (0.265893056547, 0.47737014505, 0.0602192790051, 0.0448198398943),
#             (0.31715205967, 0.240711605938, 0.0654762446139, 0.0730237819511),
#             (0.153995691063, 0.428901071989, 0.0373664030131, 0.00858452235922),
#             (0.00559367322626, 0.0791089112011, 0.00972305420369, 0.0559833278804),
#             (0.12953159662, 0.259324447733, 0.0391711915615, 0.0787194455053),
#             (0.474316734671, 0.0955731759039, 0.0307307896616, 0.0578689511311),
#             (0.227537998884, 0.119244743298, 0.0155347508731, 0.0464946287426),
#             (0.194898973577, 0.386819097308, 0.0623525386767, 0.0522080046447),
#             (0.241371292578, 0.104003854692, 0.0655300320129, 0.0280667962592),
#             (0.422975456347, 0.21982438201, 0.0552153724123, 0.0830976334031),
#             (0.211482265522, 0.456583501756, 0.0950698505041, 0.00609377148994),
#             (0.170109118018, 0.456835202197, 0.0868571426266, 0.0215892878719),
#             (0.454453661813, 0.226158304771, 0.0565930136801, 0.0696952257717),
#             (0.307520199837, 0.132233080084, 0.0427754694667, 0.0277594842489),
#             (0.00895441900238, 0.0686186289741, 0.00514014651453, 0.0067255429717),
#             (0.310777274933, 0.439954796236, 0.0101613635827, 0.0503337003359),
#             (0.143196639008, 0.241553338699, 0.0415374610938, 0.0687683126894),
#             (0.331448429615, 0.466376467798, 0.0495633161501, 0.0486257733203),
#             (0.465124007839, 0.130319672203, 0.00603682451644, 0.0329981106981),
#             (0.0627634244624, 0.453671341432, 0.0758532690511, 0.0247331822301),
#             (0.424708752036, 0.390001984289, 0.0932086515423, 0.0137084357977),
#             (0.248504617351, 0.20585489705, 0.0544422302439, 0.0290017557861),
#             (0.119995522295, 0.111772223731, 0.0803141202307, 0.047432147914),
#             (0.508864246232, 0.238470887079, 0.0538541877349, 0.0282017235677),
#             (0.204702929377, 0.402902821004, 0.0141609945434, 0.0326597038888),
#             (0.0207464387215, 0.366792429472, 0.0974149916122, 0.0889758382235)]
#         num_objects = len(params)
#         objects_all = create_metaballs(params, coeff)
    

    
        objects_mem = cl.Buffer(cfg.CTX, cl.mem_flags.READ_ONLY | \
                                cl.mem_flags.COPY_HOST_PTR, hostbuf=objects_all)
       
        ev = prg.thickness(cfg.QUEUE,
                      (n,n),
                      None,
                      thickness_mem,
                      objects_mem,
                      np.int32(num_objects),
                      vec.make_int2(0, 0),
                      vec.make_int4(0, 0, n, n),
                      cfg.NP_FLOAT(coeff),
                      cfg.NP_FLOAT(mid),
                      g_util.make_vfloat2(pixel_size.rescale(UNITS).magnitude,
                                          pixel_size.rescale(UNITS).magnitude),
                      cfg.NP_FLOAT(eps),
                      np.int32(True))
        objects_mem.release()
        
        cl.wait_for_events([ev])
        print "duration:", (ev.profile.end - ev.profile.start) * 1e-6 * q.ms
           
        res = np.empty((n,VECTOR_WIDTH*n), dtype=cfg.NP_FLOAT)
        cl.enqueue_copy(cfg.QUEUE, res, thickness_mem)
                       
        TIFF.open("/home/farago/data/thickness/radio_%04d.tif" % \
           (i + 1), "w").write_image(res[:,::VECTOR_WIDTH].astype(np.float32))
                        
    f.close()

#     ix = 877
#     iy = 1807
#         
#     data = res[iy, VECTOR_WIDTH * ix: VECTOR_WIDTH * ix + 15]
#     interval = data[:2]
#     coeffs = data[2:7]
#     print "coeffs:",
#     print_array(coeffs)
#     print "interval:",
#     print_array(interval)
#     roots = data[7:12]
#     print "thickness:", data[12]
#     print "previous:", data[13]
#     print "last accounted:", data[14]
#     print "derivative:", nr.f(nr.derivative(coeffs), roots[0])
#  
#     print
#     eps = pixel_size.rescale(q.mm).magnitude
#     print "CL roots:",
#     print_array(roots)
#     print "NR roots:",
#     print_array(nr.roots(coeffs, interval, eps))
# #     print "NP roots:",
# #     print_array(np.roots(coeffs))
#     draw(coeffs, interval)
          
#     print res[:,::VECTOR_WIDTH][iy, ix]
#     print res[:,::VECTOR_WIDTH][iy, ix+1]
    plt.figure()
    plt.imshow(res[:,::VECTOR_WIDTH], origin="lower", cmap=cm.get_cmap("gray"),
               interpolation="nearest")
    plt.colorbar()
    plt.show()
