"""Sample material represented by a complex refractive index."""
import logging
import os
import sys
import urllib
import urllib2
from HTMLParser import HTMLParser
from subprocess import Popen, PIPE
from urlparse import urljoin
import numpy as np
import quantities as q
from scipy import interpolate as interp
from syris import config as cfg, physics


LOG = logging.getLogger(__name__)


class Material(object):

    """A material represented by its *name* and *refractive_indices* calculated for *energies*."""

    def __init__(self, name, refractive_indices, energies):
        """Create material with *name* and store its complex *refractive_indices* (delta + ibeta)
        for all given *energies*.
        """
        self._name = name
        self._refractive_indices = np.array(refractive_indices)
        # To keep track which energies were used.
        self._energies = energies
        if len(self._energies) > 3:
            self._tckr = interp.splrep(self._energies, self.refractive_indices.real)
            self._tcki = interp.splrep(self._energies, self.refractive_indices.imag)

    @property
    def name(self):
        """Material *name*."""
        return self._name

    @property
    def energies(self):
        """*energies* for which the complex refractive
        index was calculated.
        """
        return self._energies

    @property
    def refractive_indices(self):
        """Get complex refractive indices (delta [phase], ibeta [absorption])
        for all energies used to create the material.
        """
        return self._refractive_indices

    def get_attenuation_coefficient(self, energy):
        """Get the linear attenuation coefficient at *energy*."""
        ref_index = self.get_refractive_index(energy)
        lam = physics.energy_to_wavelength(energy)

        return physics.ref_index_to_attenuation_coeff(ref_index, lam)

    def get_refractive_index(self, energy):
        """Interpolate refractive indices to obtain the one at *energy*."""
        if len(self.energies) < 4:
            raise MaterialError('Number of energy points \'{}\' '.format(len(self.energies)) +
                                'is too few for interpolation')
        if energy < self._energies[0] or energy > self._energies[-1]:
            raise ValueError('Energy \'{}\' not within limits \'[{}, {}]\''.
                             format(energy, self._energies[0], self._energies[-1]))
        energy = energy.rescale(self._energies.units).magnitude
        real = interp.splev(energy, self._tckr)
        imag = interp.splev(energy, self._tcki)

        return cfg.PRECISION.np_cplx(real + imag * 1j)

    def __eq__(self, other):
        return isinstance(other, Material) and self.name == other.name

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash(self.name)

    def __str__(self):
        return str(self.name)


def make_pmasf(name, energies):
    """Make a material based on the PMASF program.

    * *name* - compund name defined in "compound.dat"
    * *energies* - list of energies which will be taken
        into account [keV]
    * *steps* - number of intervals between the energies

    Return a list of refractive indices.
    """
    # determine if we are executing pmasf remotely and extract executable
    # name.
    if cfg.PMASF_FILE.startswith("ssh"):
        executable = cfg.PMASF_FILE.split()[-1]
    else:
        executable = cfg.PMASF_FILE

    cmd = "%s -C %s -E %f %f -p %d -+l%s -w Ed" % \
        (cfg.PMASF_FILE, name, energies[0].rescale(q.eV),
         energies[-1].rescale(q.eV), len(energies),
         os.path.dirname(executable))

    # Execute the pmasf program and get the text results
    # via a pipe.
    pipe = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True)
    out, err = pipe.communicate()
    if pipe.returncode != 0:
        raise RuntimeError("pmasf error (code: {0}, message: {1})".
                           format(pipe.returncode, err))

    # Parse the text output to obtain the refractive indices.
    lines = out.split("\n")
    i_0 = lines.index("# Columns: Energy[eV]\tdelta") + 1
    indices = []
    for line in lines[i_0:]:
        line = line.strip()
        if line != "":
            ref_ind = line.split("\t")[1]
            delta, beta = ref_ind.split(" ")
            indices.append(cfg.PRECISION.np_cplx(float(delta) + float(beta) * 1j))

    return Material(name, indices, energies)


def make_henke(name, energies, formula=None, density=-1):
    """Make a material *name* for *energies*, use the spcified chemical *formula* and *density*."""
    indices = _HenkeQuery(name, energies, formula=formula, density=density).refractive_indices

    return Material(name, indices, energies)


class _HenkeQuery(object):

    """Class for obtaining refractive indices obtained from `The Center For X-ray Optics`_.

    .. _The Center For X-ray Optics: http://henke.lbl.gov/optical_constants/getdb2.html
    """

    _URL = 'http://henke.lbl.gov'

    class HenkeHTMLParser(HTMLParser):

        """HTML parser for obtaining the link with refractive indices after form submission."""

        def __init__(self):
            HTMLParser.__init__(self)
            self.link = None

        def handle_starttag(self, tag, attrs):
            if attrs and attrs[0][0] == 'href':
                self.link = attrs[0][1]

    def __init__(self, name, energies, formula=None, density=-1):
        """Create material with *name* for given *energies*, use the specified *formula* and
        material *density*.
        """
        if not formula:
            formula = name
        if energies[0] < 30 * q.eV:
            raise ValueError('Minimum acceptable energy is 30 eV')
        if energies[-1] > 30 * q.keV:
            raise ValueError('Maximum acceptable energy is 30 keV')

        self.energies = energies
        self.formula = formula
        parser = _HenkeQuery.HenkeHTMLParser()

        try:
            response = self._query_server(formula, density)
            if 'error' in response.lower():
                raise MaterialError('Error looking up material `{}`'.format(name))
            parser.feed(response)
            link = urljoin(self._URL, parser.link)
            # First two lines are description
            values = urllib2.urlopen(link).readlines()[2:]
            energies_henke, indices = _parse_henke(values)
            self.refractive_indices = self._interpolate(energies_henke, indices)
        except urllib2.URLError:
            print >> sys.stderr, 'Cannot contact server, please check your Internet connection'
            raise

    def _query_server(self, formula, density):
        """Get the indices from the server."""
        data = {}
        data['Material'] = 'Enter Formula'
        data['Formula'] = formula
        data['Density'] = str(density)
        data['Scan'] = 'Energy'
        data['Min'] = str(self.energies[0].rescale(q.eV).magnitude)
        data['Max'] = str(self.energies[-1].rescale(q.eV).magnitude)
        # Get the maximum number of points and interpolate because the website output doesn't have
        # to be spaced like we want
        data['Npts'] = str(500)
        data['Output'] = 'Text File'
        url_values = urllib.urlencode(data)

        url = urljoin(self._URL, '/cgi-bin/getdb.pl')

        req = urllib2.Request(url, url_values)
        response = urllib2.urlopen(req)
        result = response.read()
        response.close()

        return result

    def _interpolate(self, energies, refractive_indices):
        """Interpolate arbitrary *energies* and *refractive_indices* into the ones
        desired by the user.
        """
        delta = refractive_indices.real
        beta = refractive_indices.imag
        desired_energies = self.energies.rescale(q.eV).magnitude

        tck = interp.splrep(energies, delta, k=1)
        idelta = interp.splev(desired_energies, tck)

        tck = interp.splrep(energies, beta, k=1)
        ibeta = interp.splev(desired_energies, tck)

        return (idelta + ibeta * 1j).astype(cfg.PRECISION.np_cplx)


def _parse_henke(response):
    """Parse server response *response* for energies, delta and beta and return a tuple
    (energies, ref_indices).
    """
    split = [line.split() for line in response]
    energies, delta, beta = zip(*split)
    delta = np.array(delta).astype(np.float)
    beta = np.array(beta).astype(np.float)
    energies = np.array(energies).astype(np.float)

    return energies, (delta + beta * 1j).astype(cfg.PRECISION.np_cplx)


class MaterialError(Exception):

    """Material errors"""

    pass
