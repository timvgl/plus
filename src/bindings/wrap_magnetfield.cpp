#include <memory>
#include <stdexcept>

#include "ferromagnet.hpp"
#include "field.hpp"
#include "fieldquantity.hpp"
#include "magnetfield.hpp"
#include "wrappers.hpp"

void wrap_magnetfield(py::module& m) {
  py::class_<MagnetField, FieldQuantity>(m, "MagnetField")
      .def(py::init([](Ferromagnet* magnet, Grid grid) {
             return std::unique_ptr<MagnetField>(new MagnetField(magnet, grid));
           }),
           py::arg("ferromagnet"), py::arg("grid"))
      .def("set_method", [](MagnetField* magnetField, std::string method) {
        if (method == "fft") {
          magnetField->setMethod(MAGNETFIELDMETHOD_FFT);
        } else if (method == "brute") {
          magnetField->setMethod(MAGNETFIELDMETHOD_BRUTE);
        } else {
          throw std::invalid_argument("Method should be \"fft\" or \"brute\"");
        }
      });
}
