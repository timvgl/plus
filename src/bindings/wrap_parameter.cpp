#include <pybind11/functional.h>
#include <pybind11/numpy.h>

#include <memory>

#include "field.hpp"
#include "fieldquantity.hpp"
#include "parameter.hpp"
#include "variable.hpp"
#include "wrappers.hpp"

void wrap_parameter(py::module& m) {
  py::class_<Parameter, FieldQuantity>(m, "Parameter")
      .def("add_time_term", py::overload_cast<const std::function<real(real)>&>(
                                &Parameter::addTimeDependentTerm))
      .def("add_time_term",
           [](Parameter* p, std::function<real(real)>& term,
              py::array_t<real> mask) {
             Field field_mask(p->system(), 1);
             setArrayInField(field_mask, mask);
             p->addTimeDependentTerm(term, field_mask);
           })
      .def_property_readonly("is_uniform", &Parameter::isUniform)
      .def_property_readonly("is_dynamic",
                             [](Parameter* p) { return p->isDynamic(); })
      .def("remove_time_terms", &Parameter::removeAllTimeDependentTerms)
      .def("set", [](Parameter* p, real value) { p->set(value); })
      .def("set", [](Parameter* p, py::array_t<real> data) {
        Field tmp(p->system(), 1);
        setArrayInField(tmp, data);
        p->set(std::move(tmp));
      });

  py::class_<FM_VectorParameter, FieldQuantity>(m, "FM_VectorParameter")
      .def(
          "add_time_term",
          [](FM_VectorParameter* p, std::function<py::array_t<real>(real)>& term) {
            auto cpp_term = [term](real t) -> real3 {
              auto np_ndarray = term(t);
              auto buffer = np_ndarray.request();

              if (buffer.ndim != 1)
                throw std::invalid_argument(
                    "Number of dimensions must be one.");

              if (buffer.size != 3)
                throw std::invalid_argument(
                    "VectorPameter value should be of size 3, got " +
                    buffer.size);

              real* ptr = static_cast<real*>(buffer.ptr);

              return real3{ptr[0], ptr[1], ptr[2]};
            };

            p->addTimeDependentTerm(cpp_term);
          })
      .def("add_time_term",
           [](FM_VectorParameter* p, std::function<py::array_t<real>(real)>& term,
              py::array_t<real> mask) {
             int ncomp = 3;
             Field field_mask(p->system(), ncomp);
             setArrayInField(field_mask, mask);

             auto cpp_term = [term](real t) -> real3 {
               auto np_ndarray = term(t);
               auto buffer = np_ndarray.request();

               if (buffer.ndim != 1)
                 throw std::invalid_argument(
                     "Number of dimensions must be one.");

               if (buffer.size != 3)
                 throw std::invalid_argument(
                     "VectorPameter value should be of size 3, got " +
                     buffer.size);

               real* ptr = static_cast<real*>(buffer.ptr);

               return real3{ptr[0], ptr[1], ptr[2]};
             };

             p->addTimeDependentTerm(cpp_term, field_mask);
           })
      .def_property_readonly("is_uniform", &FM_VectorParameter::isUniform)
      .def_property_readonly("is_dynamic",
                             [](FM_VectorParameter* p) { return p->isDynamic(); })
      .def("remove_time_terms", &FM_VectorParameter::removeAllTimeDependentTerms)
      .def("set", [](FM_VectorParameter* p, real3 value) { p->set(value); })
      .def("set", [](FM_VectorParameter* p, py::array_t<real> data) {
        Field tmp(p->system(), 3);
        setArrayInField(tmp, data);
        p->set(std::move(tmp));
      });

  py::class_<AFM_VectorParameter, FieldQuantity>(m, "AFM_VectorParameter")
      .def(
          "add_time_term",
          [](AFM_VectorParameter* p, std::function<py::array_t<real>(real)>& term) {
            auto cpp_term = [term](real t) -> real6 {
              auto np_ndarray = term(t);
              auto buffer = np_ndarray.request();

              if (buffer.ndim != 1)
                throw std::invalid_argument(
                    "Number of dimensions must be one.");

              if (buffer.size != 6)
                throw std::invalid_argument(
                    "AFM_VectorPameter value should be of size 6, got " +
                    buffer.size);

              real* ptr = static_cast<real*>(buffer.ptr);

              return real6{ptr[0], ptr[1], ptr[2], ptr[3], ptr[4], ptr[5]};
            };

            p->addTimeDependentTerm(cpp_term);
          })
      .def("add_time_term",
           [](AFM_VectorParameter* p, std::function<py::array_t<real>(real)>& term,
              py::array_t<real> mask) {
             int ncomp = 6;
             Field field_mask(p->system(), ncomp);
             setArrayInField(field_mask, mask);

             auto cpp_term = [term](real t) -> real6 {
               auto np_ndarray = term(t);
               auto buffer = np_ndarray.request();

               if (buffer.ndim != 1)
                 throw std::invalid_argument(
                     "Number of dimensions must be one.");

               if (buffer.size != 6)
                 throw std::invalid_argument(
                     "AFM_VectorPameter value should be of size 6, got " +
                     buffer.size);

               real* ptr = static_cast<real*>(buffer.ptr);

               return real6{ptr[0], ptr[1], ptr[2], ptr[3], ptr[4], ptr[5]};
             };

             p->addTimeDependentTerm(cpp_term, field_mask);
           })
      .def_property_readonly("is_uniform", &AFM_VectorParameter::isUniform)
      .def_property_readonly("is_dynamic",
                             [](AFM_VectorParameter* p) { return p->isDynamic(); })
      .def("remove_time_terms", &AFM_VectorParameter::removeAllTimeDependentTerms)
      .def("set", [](AFM_VectorParameter* p, real6 value) { p->set(value); })
      .def("set", [](AFM_VectorParameter* p, py::array_t<real> data) {
        Field tmp(p->system(), 6);
        setArrayInField(tmp, data);
        p->set(std::move(tmp));
      });
}
