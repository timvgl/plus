#include <memory>

#include "magnetfieldkernel.hpp"
#include "ferromagnet.hpp"
#include "parameter.hpp"
#include "world.hpp"
#include "wrappers.hpp"

void wrap_ferromagnet(py::module& m) {
  py::class_<Ferromagnet>(m, "Ferromagnet")
      .def_property_readonly("name", &Ferromagnet::name)
      .def_property_readonly("grid", &Ferromagnet::grid)

      .def_property("magnetization", &Ferromagnet::magnetization,
                    [](Ferromagnet* fm, py::object data) {
                      py::cast(fm->magnetization()).attr("set")(data);
                    })

      .def_property(
          "msat", [](const Ferromagnet* fm) { return &fm->msat; },
          [](Ferromagnet* fm, py::object data) {
            py::cast(&fm->msat).attr("set")(data);
          })

      .def_property(
          "alpha", [](const Ferromagnet* fm) { return &fm->alpha; },
          [](Ferromagnet* fm, py::object data) {
            py::cast(&fm->alpha).attr("set")(data);
          })

      .def_property(
          "ku1", [](const Ferromagnet* fm) { return &fm->ku1; },
          [](Ferromagnet* fm, py::object data) {
            py::cast(&fm->ku1).attr("set")(data);
          })

      .def_property(
          "aex", [](const Ferromagnet* fm) { return &fm->aex; },
          [](Ferromagnet* fm, py::object data) {
            py::cast(&fm->aex).attr("set")(data);
          })

      .def_property(
          "anisU", [](const Ferromagnet* fm) { return &fm->anisU; },
          [](Ferromagnet* fm, py::object data) {
            py::cast(&fm->anisU).attr("set")(data);
          })

      .def_readwrite("enable_demag", &Ferromagnet::enableDemag)

      .def_property_readonly("demag_field", &Ferromagnet::demagField)
      .def_property_readonly("demag_energy_density",
                             &Ferromagnet::demagEnergyDensity)
      .def_property_readonly("demag_energy",
                             &Ferromagnet::demagEnergy)

      .def_property_readonly("anisotropy_field", &Ferromagnet::anisotropyField)
      .def_property_readonly("anisotropy_energy_density",
                             &Ferromagnet::anisotropyEnergyDensity)
      .def_property_readonly("anisotropy_energy",
                             &Ferromagnet::anisotropyEnergy)

      .def_property_readonly("exchange_field", &Ferromagnet::exchangeField)
      .def_property_readonly("exchange_energy_density",
                             &Ferromagnet::exchangeEnergyDensity)
      .def_property_readonly("exchange_energy",
                             &Ferromagnet::exchangeEnergy)

      .def_property_readonly("external_field", &Ferromagnet::externalField)
      .def_property_readonly("zeeman_energy_density",
                             &Ferromagnet::zeemanEnergyDensity)
      .def_property_readonly("zeeman_energy",
                             &Ferromagnet::zeemanEnergy)

      .def_property_readonly("effective_field", &Ferromagnet::effectiveField)
      .def_property_readonly("total_energy_density", &Ferromagnet::totalEnergyDensity)
      .def_property_readonly("total_energy", &Ferromagnet::totalEnergy)

      .def_property_readonly("torque", &Ferromagnet::torque)

      .def("magnetic_field_from_magnet", 
           [](const Ferromagnet* fm, Ferromagnet* magnet) {
             return (FieldQuantity*)(fm->getMagnetField(magnet));
           }, py::return_value_policy::reference)

      .def("minimize", &Ferromagnet::minimize, py::arg("tol") = 1e-6,
           py::arg("nsamples") = 10)

      // TODO: remove demagkernel function
      .def("_demagkernel",
           [](const Ferromagnet* fm) {
             Grid grid = fm->grid();
             real3 cellsize = fm->world()->cellsize();
             MagnetFieldKernel demagKernel(grid, grid, cellsize);
             std::unique_ptr<Field> kernel(
                 new Field(demagKernel.field()->grid(), 6));
             kernel.get()->copyFrom(demagKernel.field());
             return fieldToArray(kernel.get());
           })

      //.def("__repr__", [](const Ferromagnet &f) {
      //  return "Ferromagnet named '" + f.name() + "'";
      //})
      ;
}