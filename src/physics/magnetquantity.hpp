#pragma once

#include<memory>

#include"grid.hpp"
#include"ferromagnet.hpp"


class MagnetFieldQuantity {
  public:
    MagnetFieldQuantity(Ferromagnet * magnet) : magnet_(magnet) {}
  private:
    std::weak_ptr<Ferromagnet> magnet_;
};