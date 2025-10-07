#pragma once

#include "quantityevaluator.hpp"

class Antiferromagnet;
class Ferromagnet;
class Field;
class NcAfm;

Field evalFMFullMag(const Ferromagnet* magnet);
Field evalHMFullMag(const HostMagnet* magnet);
Field evalHMFullMagOn(const HostMagnet* magnet, cudaStream_t stream_);


FM_FieldQuantity fullMagnetizationQuantity(const Ferromagnet*);
AFM_FieldQuantity fullMagnetizationQuantity(const Antiferromagnet*);
NcAfm_FieldQuantity fullMagnetizationQuantity(const NcAfm*);