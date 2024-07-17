#include <memory>
#include <stdexcept>

#include "antiferromagnet.hpp"
#include "fieldquantity.hpp"
#include "mumaxworld.hpp"
#include "neel.hpp"
#include "parameter.hpp"
#include "totalmag.hpp"
#include "world.hpp"
#include "wrappers.hpp"

void wrap_antiferromagnet(py::module& m) {
  py::class_<Antiferromagnet>(m, "Antiferromagnet")
      .def_property_readonly("name", &Antiferromagnet::name)
      .def_property_readonly("system", &Antiferromagnet::system)
      .def_property_readonly("world", &Antiferromagnet::mumaxWorld)
      .def("sub1", &Antiferromagnet::sub1, py::return_value_policy::reference)
      .def("sub2", &Antiferromagnet::sub2, py::return_value_policy::reference)
      .def("sublattices", &Antiferromagnet::sublattices, py::return_value_policy::reference)
      .def_readonly("afmex_cell", &Antiferromagnet::afmex_cell)
      .def_readonly("afmex_nn", &Antiferromagnet::afmex_nn)
      .def_readonly("latcon", &Antiferromagnet::latcon)
      .def("minimize", &Antiferromagnet::minimize, py::arg("tol") = 1e-6,
           py::arg("nsamples") = 20)
      .def("relax", &Antiferromagnet::relax, py::arg("tol"));
      
  m.def("neel_vector", &neelVectorQuantity);
  m.def("total_magnetization", &totalMagnetizationQuantity);
}