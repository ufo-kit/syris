from syris import config as cfg
import logging
import os
from subprocess import Popen, PIPE
import quantities as q


logger = logging.getLogger(__name__)
pmasf_binary = "pmasf"

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
        return self._name
    
    @property
    def energies(self):
        return self._energies
    
    @property
    def refractive_indices(self):
        """Get complex refractive indices (delta [phase], ibeta [absorption])
        for all energies used to create the material.
        """
        return self._refractive_indices
    
    def pack(self):
        """Pack material into an OpenCL compliant structure."""
        raise NotImplementedError
            
        
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
            cmd = "%s -C %s -E %f %f -p %d -+l%s -w Ed" %\
                (pmasf_binary, name, energies[0].rescale(q.eV),
                 energies[-1].rescale(q.eV), len(energies),
                 os.path.dirname(pmasf_binary))
                
            # Execute the pmasf program and get the text results
            # via a pipe.
            p = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True)
            out, err = p.communicate()
            if p.returncode != 0:
                raise RuntimeError("pmasf error (code: {0}, message: {1})".\
                                   format(p.returncode, err))

            # Parse the text output to obtain the refractive indices.
            lines = out.split("\n")
            i0 = lines.index("# Columns: Energy[eV]\tdelta")+1
            for line in lines[i0:]:
                line = line.strip()
                if line != "":
                    ref_ind = line.split("\t")[1]
                    delta, beta = ref_ind.split(" ")
                    self._refractive_indices.append(\
                        cfg.np_cplx(cfg.np_float(delta) +
                                    cfg.np_float(beta)*1j))
