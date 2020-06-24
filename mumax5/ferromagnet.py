import _mumax5cpp as _cpp


class Ferromagnet:
    """Ferromagnet"""

    def __init__(self, world, grid, name=""):
        self._impl = world._impl.add_ferromagnet(grid, name)

    @property
    def name(self):
        return self._impl.name

    @property
    def grid(self):
        return self._impl.grid

    @property
    def world(self):
        return self._impl.world

    @property
    def cellsize(self):
        return self._impl.cellsize

    @property
    def magnetization(self):
        return self._impl.magnetization

    @magnetization.setter
    def magnetization(self, value):
        self._impl.magnetization.set(value)

    @property
    def enable_demag(self):
        return self._impl.enable_demag

    @enable_demag.setter
    def enable_demag(self, value):
        self._impl.enable_demag = value

    def minimize(self):
        self._impl.minimize()

    #----- MATERIAL PARAMETERS -----------

    @property
    def msat(self):
        return self._impl.msat

    @msat.setter
    def msat(self, value):
        self._impl.msat.set(value)

    @property
    def alpha(self):
        return self._impl.alpha

    @alpha.setter
    def alpha(self, value):
        self._impl.alpha.set(value)

    @property
    def aex(self):
        return self._impl.aex

    @aex.setter
    def aex(self, value):
        self._impl.aex.set(value)

    @property
    def ku1(self):
        return self._impl.ku1

    @ku1.setter
    def ku1(self, value):
        self._impl.ku1.set(value)

    @property
    def ku2(self):
        return self._impl.ku2

    @ku2.setter
    def ku2(self, value):
        self._impl.ku2.set(value)

    @property
    def anisU(self):
        return self._impl.anisU

    @anisU.setter
    def anisU(self, value):
        self._impl.anisU.set(value)

    @property
    def idmi(self):
        return self._impl.idmi

    @idmi.setter
    def idmi(self, value):
        self._impl.idmi.set(value)

    @property
    def xi(self):
        return self._impl.xi

    @xi.setter
    def xi(self, value):
        self._impl.xi.set(value)

    @property
    def pol(self):
        return self._impl.pol

    @pol.setter
    def pol(self, value):
        self._impl.pol.set(value)

    @property
    def jcur(self):
        return self._impl.jcur

    @jcur.setter
    def jcur(self, value):
        self._impl.jcur.set(value)

    @property
    def temperature(self):
        return self._impl.temperature

    @temperature.setter
    def temperature(self, value):
        self._impl.temperature.set(value)

    #----- QUANTITIES ----------------------

    @property
    def torque(self):
        return _cpp.torque(self._impl)

    @property
    def llg_torque(self):
        return _cpp.llg_torque(self._impl)

    @property
    def spin_transfer_torque(self):
        return _cpp.spin_transfer_torque(self._impl)

    @property
    def demag_field(self):
        return _cpp.demag_field(self._impl)

    @property
    def demag_energy_density(self):
        return _cpp.demag_energy_density(self._impl)

    @property
    def demag_energy(self):
        return _cpp.demag_energy(self._impl)

    @property
    def anisotropy_field(self):
        return _cpp.anisotropy_field(self._impl)

    @property
    def anisotropy_energy_density(self):
        return _cpp.anisotropy_energy_density(self._impl)

    @property
    def anisotropy_energy(self):
        return _cpp.anisotropy_energy(self._impl)
  
    @property
    def exchange_field(self):
        return _cpp.exchange_field(self._impl)

    @property
    def exchange_energy_density(self):
        return _cpp.exchange_energy_density(self._impl)

    @property
    def exchange_energy(self):
        return _cpp.exchange_energy(self._impl)

    @property
    def max_angle(self):
        return _cpp.max_angle(self._impl) 
  
    @property
    def interfacialdmi_field(self):
        return _cpp.interfacialdmi_field(self._impl)

    @property
    def interfacialdmi_energy_density(self):
        return _cpp.interfacialdmi_energy_density(self._impl)

    @property
    def interfacialdmi_energy(self):
        return _cpp.interfacialdmi_energy(self._impl)

    @property
    def external_field(self):
        return _cpp.external_field(self._impl)

    @property
    def zeeman_energy_density(self):
        return _cpp.zeeman_energy_density(self._impl)

    @property
    def zeeman_energy(self):
        return _cpp.zeeman_energy(self._impl)

    @property
    def effective_field(self):
        return _cpp.effective_field(self._impl)

    @property
    def total_energy_density(self):
        return _cpp.total_energy_density(self._impl)

    @property
    def total_energy(self):
        return _cpp.total_energy(self._impl)
