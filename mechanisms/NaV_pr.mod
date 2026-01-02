NEURON {
    SUFFIX NaV_pr
    USEION na READ ena WRITE ina
    RANGE gMax, mVHalf, mVWidth, hVHalf, hVWidth, mTau, hTau
}

UNITS {
    (mS) = (millisiemens)
    (mV) = (millivolt)
    (mA) = (milliamp)
    (S)  = (siemens)
}

PARAMETER {
    gMax     = 0.5 (mS/cm2)   : peak conductance density

    : activation
    mVHalf   = -48 (mV)       : half-activation voltage
    mVWidth  =   5 (mV)       : activation slope
    mTauMult = 0.01 (ms)      : activation time constant

    : inactivation
    hVHalf   = -49.5 (mV)     : half-inactivation voltage
    hVWidth  =    2 (mV)      : inactivation slope
    hTauMult = 0.5 (ms)       : inactivation time constant
}

ASSIGNED {
    v    (mV)
    ena  (mV)
    ina  (mA/cm2)
    gNaV (mS/cm2)
    mss                : steady-state activation
    mTau (ms)
    hss                : steady-state inactivation
    hTau (ms)
}

STATE { m h }

BREAKPOINT {
    SOLVE states METHOD cnexp
    gNaV = gMax * m*m*m * h
    ina  = gNaV * (v - ena)
}

DERIVATIVE states {
    rates()
    m' = (mss - m) / mTau
    h' = (hss - h) / hTau
}

INITIAL {
    rates()
    m = mss
    h = hss
}

PROCEDURE rates() {
    : Boltzmann steady-state curves
    mss  = 1 / (1 + exp((mVHalf - v) / mVWidth))
    hss  = 1 / (1 + exp((v - hVHalf) / hVWidth))

    : constant time-constants
    mTau = mTauMult
    hTau = hTauMult
}
