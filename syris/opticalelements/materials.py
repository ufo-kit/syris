"""Sample material represented by a complex refractive index."""
from syris import config as cfg, physics
import logging
import os
from subprocess import Popen, PIPE
import quantities as q


LOGGER = logging.getLogger(__name__)


class Material(object):

    """Abstract class representing materials."""

    def __init__(self, name, energies):
        """Create material with *name* store its complex refractive indices
        for all given *energies*.
        """
        self._name = name
        self._refractive_indices = []
        # To keep track which energies were used.
        self._energies = energies

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

    def get_attenuation_coeff(self, energy_index):
        """Get the linear attenuation coefficient baseo on *energy_index*
        to the energies for which the material was defined.
        """
        return physics.ref_index_to_attenuation(
            self.refractive_indices[energy_index],
            self.energies[energy_index])


class PMASFMaterial(Material):

    """Class representing materials based on their complex refractive
    indices calculated by PMASF program written by Petr Mikulik.
    """

    def __init__(self, name, energies):
        """Create material with *name* for given *energies*."""
        Material.__init__(self, name, energies)
        self._create_refractive_indices(name, energies)

    def _create_refractive_indices(self, name, energies):
        """Calculate refractive indices with pmasf program, where

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
        for line in lines[i_0:]:
            line = line.strip()
            if line != "":
                ref_ind = line.split("\t")[1]
                delta, beta = ref_ind.split(" ")
                self._refractive_indices.append(
                    cfg.NP_CPLX(cfg.NP_FLOAT(delta) +
                                cfg.NP_FLOAT(beta) * 1j))
