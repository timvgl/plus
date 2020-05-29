#pragma once

#include<memory>

#include"grid.hpp"
#include"ferromagnet.hpp"


class FM_FieldQuantity {
  public:
    FM_FieldQuantity(Ferromagnet * magnet) : magnet_(magnet) {}
  private:
    std::weak_ptr<Ferromagnet> magnet_;
};