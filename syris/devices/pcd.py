from syris.devices.cameras import Camera
from syris.physics import energy_to_wavelength, wavelength_to_energy
import quantities as q
from syris import config as cfg
import numpy as np
import scipy.interpolate as interp
from syris.imageprocessing import decimate
from scipy.special import erf
import syris.gpu.util as gutil


wavelengths_Si = energy_to_wavelength(np.array((1.00000E-03, 1.50000E-03, 1.83890E-03, 1.83890E-03, 2.00000E-03, 3.00000E-03, 4.00000E-03, 5.00000E-03, 6.00000E-03, 8.00000E-03, 1.00000E-02, 1.50000E-02, 2.00000E-02, 3.00000E-02, 4.00000E-02, 5.00000E-02, 6.00000E-02, 8.00000E-02, 1.00000E-01, 1.50000E-01, 2.00000E-01, 3.00000E-01, 4.00000E-01, 5.00000E-01, 6.00000E-01, 8.00000E-01, 1.00000E+00, 1.25000E+00, 1.50000E+00, 2.00000E+00, 3.00000E+00, 4.00000E+00, 5.00000E+00, 6.00000E+00, 8.00000E+00, 1.00000E+01, 1.50000E+01, 2.00000E+01))*q.MeV)

mu_Si = np.array((1.570E+03, 5.355E+02, 3.092E+02, 3.192E+03, 2.777E+03, 9.784E+02, 4.529E+02, 2.450E+02, 1.470E+02, 6.468E+01, 3.389E+01, 1.034E+01, 4.464E+00, 1.436E+00, 7.012E-01, 4.385E-01, 3.207E-01, 2.228E-01, 1.835E-01, 1.448E-01, 1.275E-01, 1.082E-01, 9.614E-02, 8.748E-02, 8.077E-02, 7.082E-02, 6.361E-02, 5.688E-02, 5.183E-02, 4.480E-02, 3.678E-02, 3.240E-02, 2.967E-02, 2.788E-02, 2.574E-02, 2.462E-02, 2.352E-02, 2.338E-02))*2.336/q.cm

wavelengths_GaAs= energy_to_wavelength(np.array((1.00000E-03, 1.05613E-03, 1.11540E-03, 1.11540E-03, 1.12877E-03, 1.14230E-03, 1.14230E-03, 1.21752E-03, 1.29770E-03, 1.29770E-03, 1.31034E-03, 1.32310E-03, 1.32310E-03, 1.34073E-03, 1.35860E-03, 1.35860E-03, 1.50000E-03, 1.52650E-03, 1.52650E-03, 2.00000E-03, 3.00000E-03, 4.00000E-03, 5.00000E-03, 6.00000E-03, 8.00000E-03, 1.00000E-02, 1.03671E-02, 1.03671E-02, 1.10916E-02, 1.18667E-02, 1.18667E-02, 1.50000E-02, 2.00000E-02, 3.00000E-02, 4.00000E-02, 5.00000E-02, 6.00000E-02, 8.00000E-02, 1.00000E-01, 1.50000E-01, 2.00000E-01, 3.00000E-01, 4.00000E-01, 5.00000E-01, 6.00000E-01, 8.00000E-01, 1.00000E+00, 1.25000E+00, 1.50000E+00, 2.00000E+00, 3.00000E+00, 4.00000E+00, 5.00000E+00, 6.00000E+00, 8.00000E+00, 1.00000E+01, 1.50000E+01, 2.00000E+01))*q.MeV)

mu_GaAs = np.array((1.917E+03, 1.685E+03, 1.481E+03, 2.772E+03, 3.180E+03, 3.532E+03, 4.372E+03, 4.130E+03, 3.657E+03, 4.066E+03, 3.971E+03, 3.879E+03, 5.652E+03, 5.525E+03, 5.415E+03, 6.266E+03, 5.159E+03, 4.939E+03, 5.278E+03, 2.731E+03, 9.702E+02, 4.539E+02, 2.495E+02, 1.524E+02, 6.960E+01, 3.780E+01, 3.425E+01, 1.260E+02, 1.059E+02, 8.899E+01, 1.685E+02, 9.220E+01, 4.258E+01, 1.397E+01, 6.262E+00, 3.365E+00, 2.042E+00, 9.587E-01, 5.598E-01, 2.509E-01, 1.671E-01, 1.137E-01, 9.371E-02, 8.248E-02, 7.484E-02, 6.452E-02, 5.751E-02, 5.122E-02, 4.676E-02, 4.102E-02, 3.538E-02, 3.288E-02, 3.172E-02, 3.121E-02, 3.117E-02, 3.170E-02, 3.354E-02, 3.543E-02))*5.32/q.cm


class TimepixHexa(Camera):
    def __init__(self, wavelengths=wavelengths_GaAs, mu=mu_GaAs,sensor_thickness=500*q.um, threshold=7*q.keV, exp_time=1*q.s, psf_sigma = 20*q.um, erf_sigma=1.5*q.keV, thl_dispersion=1.*q.keV, fixed_dispersion=True):
        self._threshold = threshold
        self.pixel_size = 55*q.um
        self._max_grey_value = 11810
        self.dtype = int
        self.shape = (512, 768)
        self.psf_sigma = psf_sigma
        self.erf_sigma = erf_sigma
        self._fixed_dispersion = fixed_dispersion
        self.thl_dispersion = thl_dispersion

        self.mu = mu
        self._wavelengths = wavelengths
        self.sensor_thickness = sensor_thickness
        self._lowest_threshold = 6.999*q.keV

    @property
    def thl(self):
        return self._threshold

    @property
    def max_grey_value(self):
        return self._max_grey_value

    @thl.setter
    def thl(self, th):
        if th > self._lowest_threshold:
            self._threshold = th
        else:
            raise(Exception("Threshold can not be set below %s"%self._lowest_threshold))

    @property
    def thl_dispersion(self):
        return self._thl_dispersion

    @thl_dispersion.setter
    def thl_dispersion(self, disp):
        self._thl_dispersion = disp
        if self._fixed_dispersion:
            np.random.seed(0)
        self._threshold_shift = np.random.randn(self.shape[0], self.shape[1])*self._thl_dispersion

    def get_image(self, photons, wavelength, photons_pixel_size, queue=None):
        if queue is None:
            queue = cfg.OPENCL.queue
        energy = wavelength_to_energy(wavelength)

        e = energy.rescale(q.keV).magnitude
        e_i = wavelength_to_energy(self._wavelengths).rescale(q.keV).magnitude
        mu_i = self.mu.rescale(1/q.mm).magnitude
        mu = np.exp(np.interp(np.log(e), np.log(e_i), np.log(mu_i)))/q.mm

        p = np.copy(photons)
        efficiency = 1-np.exp((-mu*self.sensor_thickness).simplified)
        p *= efficiency
        p = np.round(p)

        s = self.psf_sigma/self.pixel_size
        p = np.round(gutil.get_host(decimate(p.astype(np.int), self.shape, s, queue=queue, block=True))).astype(np.int)

        thl = self._threshold + self._threshold_shift
        thl = thl.rescale(q.keV).magnitude
        energy = energy.rescale(q.keV).magnitude
        sigma = self.erf_sigma.rescale(q.keV).magnitude
        probability = 1 - 0.5 * (1 + erf( ((thl-energy) / (sigma * np.sqrt(2) )) ) )

        img = np.zeros_like(p)
        #photon noise
        img[p>=1] = np.random.poisson(p[p>=1])
        # threshold-noise
        img[img>=1] = np.random.binomial(img[img>=1], probability[img>=1])
        img[img>=self.max_grey_value] = self.max_grey_value
        return img
