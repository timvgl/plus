#include <memory>
#include <stdexcept>

#include "ferromagnet.hpp"
#include "fieldquantity.hpp"
#include "magnetfieldkernel.hpp"
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

      .def_property(
          "temperature", [](const Ferromagnet* fm) { return &fm->msat; },
          [](Ferromagnet* fm, py::object data) {
            py::cast(&fm->temperature).attr("set")(data);
          })

      .def_readwrite("enable_demag", &Ferromagnet::enableDemag)

      .def_property_readonly("demag_field",
                             [](const Ferromagnet* fm) {
                               return std::unique_ptr<FieldQuantity>(
                                   new DemagField(fm->getHandle()));
                             })

      .def_property_readonly("demag_energy_density",
                             [](const Ferromagnet* fm) {
                               return std::unique_ptr<FieldQuantity>(
                                   new DemagEnergyDensity(fm->getHandle()));
                             })

      .def_property_readonly("demag_energy",
                             [](const Ferromagnet* fm) {
                               return std::unique_ptr<ScalarQuantity>(
                                   new DemagEnergy(fm->getHandle()));
                             })

      .def_property_readonly("anisotropy_field",
                             [](const Ferromagnet* fm) {
                               return std::unique_ptr<FieldQuantity>(
                                   new AnisotropyField(fm->getHandle()));
                             })

      .def_property_readonly(
          "anisotropy_energy_density",
          [](const Ferromagnet* fm) {
            return std::unique_ptr<FieldQuantity>(
                new AnisotropyEnergyDensity(fm->getHandle()));
          })

      .def_property_readonly("anisotropy_energy",
                             [](const Ferromagnet* fm) {
                               return std::unique_ptr<ScalarQuantity>(
                                   new AnisotropyEnergy(fm->getHandle()));
                             })

      .def_property_readonly("exchange_field",
                             [](const Ferromagnet* fm) {
                               return std::unique_ptr<FieldQuantity>(
                                   new ExchangeField(fm->getHandle()));
                             })

      .def_property_readonly("exchange_energy_density",
                             [](const Ferromagnet* fm) {
                               return std::unique_ptr<FieldQuantity>(
                                   new ExchangeEnergyDensity(fm->getHandle()));
                             })

      .def_property_readonly("exchange_energy",
                             [](const Ferromagnet* fm) {
                               return std::unique_ptr<ScalarQuantity>(
                                   new ExchangeEnergy(fm->getHandle()));
                             })

      .def_property_readonly("external_field",
                             [](const Ferromagnet* fm) {
                               return std::unique_ptr<FieldQuantity>(
                                   new ExternalField(fm->getHandle()));
                             })

      .def_property_readonly("zeeman_energy_density",
                             [](const Ferromagnet* fm) {
                               return std::unique_ptr<FieldQuantity>(
                                   new ZeemanEnergyDensity(fm->getHandle()));
                             })

      .def_property_readonly("zeeman_energy",
                             [](const Ferromagnet* fm) {
                               return std::unique_ptr<ScalarQuantity>(
                                   new ZeemanEnergy(fm->getHandle()));
                             })

      .def_property_readonly("effective_field",
                             [](const Ferromagnet* fm) {
                               return std::unique_ptr<FieldQuantity>(
                                   new EffectiveField(fm->getHandle()));
                             })

      .def_property_readonly("total_energy_density",
                             [](const Ferromagnet* fm) {
                               return std::unique_ptr<FieldQuantity>(
                                   new TotalEnergyDensity(fm->getHandle()));
                             })

      .def_property_readonly("total_energy",
                             [](const Ferromagnet* fm) {
                               return std::unique_ptr<ScalarQuantity>(
                                   new TotalEnergy(fm->getHandle()));
                             })

      .def_property_readonly(
          "torque",
          [](const Ferromagnet* fm) {
            return std::unique_ptr<FieldQuantity>(new Torque(fm->getHandle()));
          })

      .def_property_readonly("thermal_noise",
                             [](const Ferromagnet* fm) {
                               return std::unique_ptr<FieldQuantity>(
                                   new ThermalNoise(fm->getHandle()));
                             })

      .def(
          "magnetic_field_from_magnet",
          [](const Ferromagnet* fm, Ferromagnet* magnet) {
            const MagnetField* magnetField =
                fm->getMagnetField(magnet->getHandle());
            if (!magnetField)
              throw std::runtime_error(
                  "Can not compute the magnetic field of the magnet");
            return magnetField;
          },
          py::return_value_policy::reference)

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