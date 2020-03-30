#include"wrappers.hpp"
#include"ferromagnet.hpp"

#include<iostream>

void wrap_ferromagnet(py::module& m) {

    py::class_<Ferromagnet>(m, "Ferromagnet")
        .def("name", &Ferromagnet::name)
        .def("grid", &Ferromagnet::grid)
        .def_property_readonly("magnetization", &Ferromagnet::magnetization)
        .def_readwrite("msat", &Ferromagnet::msat)
        .def_readwrite("alpha", &Ferromagnet::alpha)
        .def_readwrite("ku1", &Ferromagnet::ku1)
        .def_readwrite("anisU", &Ferromagnet::anisU)
        .def_readwrite("aex", &Ferromagnet::aex)
        .def_property_readonly("anisotropy_field", &Ferromagnet::anisotropyField)
        .def_property_readonly("exchange_field", &Ferromagnet::exchangeField)
        .def_property_readonly("effective_field", &Ferromagnet::effectiveField)
        .def_property_readonly("torque", &Ferromagnet::torque)

        //.def("__repr__", [](const Ferromagnet &f) {
        //  return "Ferromagnet named '" + f.name() + "'";
        //})
        ;
}