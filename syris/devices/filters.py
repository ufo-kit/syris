"""Module for beam filters which cause light attenuation. Filters are assumed
to be homogeneous, thus no phase change effects are introduced when
a wavefield passes through them.
"""


class Filter(object):

    """Beam frequency filter."""

    def __init__(self, thickness, material):
        """Create a beam filter with projected *thickness* in beam direction
        and *material*.
        """
        self.thickness = thickness.simplified
        self.material = material

    def get_attenuation(self, energy_index):
        """Get attenuation based on *energy_index*, which is an index
        into energies for which the material was created.
        """
        return self.thickness * \
            self.material.get_attenuation_coefficient(energy_index)


class Scintillator(Filter):

    """Scintillator emits visible light when it is irradiated by X-rays."""

    def __init__(self, thickness, material, light_yields,
                 quantuim_efficiencies, optical_ref_index):
        """Create a scintillator with *light_yields* as a tuple of (energies,
        light_yield), defining the light yield for given energy.
        *quantuim_efficiencies* is a tuple (energy, quantum_efficiency) which
        gives us the quantum efficiency for visible light energy.
        *optical_ref_index* is the refractive index between the scintillator
        material and air."""
        super(Scintillator, self).__init__(thickness, material)
        self.ligh_yields = light_yields
        self.quantum_effs = quantuim_efficiencies
        self.opt_ref_index = optical_ref_index
