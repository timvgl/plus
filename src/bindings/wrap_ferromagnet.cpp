#include <memory>

#include "demagkernel.hpp"
#include "ferromagnet.hpp"
#include "world.hpp"
#include "wrappers.hpp"

void wrap_ferromagnet(py::module& m) {
  py::class_<Ferromagnet>(m, "Ferromagnet")
      .def_property_readonly("name", &Ferromagnet::name)
      .def_property_readonly("grid", &Ferromagnet::grid)
      .def_property_readonly("magnetization", &Ferromagnet::magnetization)
      .def_readwrite("msat", &Ferromagnet::msat)
      .def_readwrite("alpha", &Ferromagnet::alpha)
      .def_readonly("ku1", &Ferromagnet::ku1)
      .def_readwrite("anisU", &Ferromagnet::anisU)
      .def_readwrite("aex", &Ferromagnet::aex)
      .def_readwrite("enable_demag", &Ferromagnet::enableDemag)
      .def_property_readonly("demag_field", &Ferromagnet::demagField)
      .def_property_readonly("anisotropy_field", &Ferromagnet::anisotropyField)
      .def_property_readonly("exchange_field", &Ferromagnet::exchangeField)
      .def_property_readonly("effective_field", &Ferromagnet::effectiveField)
      .def_property_readonly("torque", &Ferromagnet::torque)
      .def("minimize", &Ferromagnet::minimize, py::arg("tol") = 1e-6,
           py::arg("nsamples") = 10)

      // TODO: remove demagkernel function
      .def("_demagkernel",
           [](const Ferromagnet* fm) {
             Grid grid = fm->grid();
             real3 cellsize = fm->world()->cellsize();
             DemagKernel demagKernel(grid, grid, cellsize);
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