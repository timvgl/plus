#pragma once

#include "antiferromagnetquantity.hpp"
#include "ferromagnetquantity.hpp"

class Antiferromagnet;
class Ferromagnet;
class Field;

Field evalTorque(const Ferromagnet*);
Field evalAFMTorque(const Antiferromagnet*, const Ferromagnet*);

Field evalLlgTorque(const Ferromagnet*);
Field evalAFMLlgTorque(const Antiferromagnet*, const Ferromagnet*);

Field evalRelaxTorque(const Ferromagnet*);
Field evalAFMRelaxTorque(const Antiferromagnet*, const Ferromagnet*);

FM_FieldQuantity torqueQuantity(const Ferromagnet*);
AFM_FieldQuantity AFM_torqueQuantity(const Antiferromagnet*, const Ferromagnet*);

FM_FieldQuantity llgTorqueQuantity(const Ferromagnet*);
AFM_FieldQuantity AFM_llgTorqueQuantity(const Antiferromagnet*, const Ferromagnet*);

FM_FieldQuantity relaxTorqueQuantity(const Ferromagnet*);
AFM_FieldQuantity AFM_relaxTorqueQuantity(const Antiferromagnet*, const Ferromagnet*);