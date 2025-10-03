# firsttry.py — Hämatit in mumax⁺, zeitgetaktetes run() + Live-Stream
from analyze_GPU import GPUUtilMonitor
with GPUUtilMonitor(interval=1) as mon:
    from mumaxplus import World, Grid, Antiferromagnet
    import math

    # ------- Sim-Parameter (leicht editierbar) -------
    PHASE        = "easy_axis"    # "easy_axis" oder "easy_plane"
    Lx, Ly, t    = 1e-6, 0.5e-6, 20e-9
    nx, ny, nz   = 256, 128, 1
    runtime      = 3e-10          # Gesamtlaufzeit [s]
    sample_every = 1e-10          # pro Tick laufende Sim-Zeit [s]

    # ------- Wir machen das Setup NACH dem Laden der Seite -------

    # ---- Physik-Setup ----
    cx, cy, cz = Lx/nx, Ly/ny, t/nz
    grid  = Grid((nx, ny, nz))
    world = World((cx, cy, cz), pbc_repetitions=(0, 0, 0))  # mastergrid weggelassen
    afm   = Antiferromagnet(world, grid)
    sub1, sub2 = afm.sub1, afm.sub2

    mu0   = 4e-7*math.pi
    Ms    = 3.31e6
    A_F   = 4.4e-12
    H2par = +24e-3 if PHASE=="easy_axis" else -24e-3
    H6    = 137e-6
    Hex_T = 1040.0
    HDMI  = 2.75
    a_lat = 5.067e-10
    afm.latcon = a_lat

    for s in (sub1, sub2):
        s.msat = Ms
        s.alpha = 0.005
        s.aex = A_F

    Ku1 = 0.5*mu0*Ms*H2par
    for s in (sub1, sub2):
        s.ku1 = Ku1
        s.anisU = (0,0,1)

    if PHASE == "easy_plane":
        eps = 0.12 * (0.5*mu0*Ms*H6)
        for k in range(6):
            phi = 2*math.pi*k/6.0
            dir2d = (math.cos(phi), math.sin(phi), 0.0)
            sub1.ku1 += eps; sub1.anisU = dir2d
            sub2.ku1 += eps; sub2.anisU = dir2d

    # AFM-Kopplung (negativ!)
    afm.afmex_cell = - mu0 * Ms * Hex_T * (a_lat**2)

    # homogene DMI (Canting)
    D_hom = 0.25 * (mu0 * Ms * HDMI)
    afm.dmi_vector = (0.0, 0.0, D_hom)

    # Startzustand & Bias
    sub1.magnetization = (0,0,+1)
    sub2.magnetization = (0,0,-1)
    afm.bias_magnetic_field = (1e-4, -1e-4, 0.0)

    # Anregung (Sinc)
    Bac, fmax = 2e-3, 2.0e11
    world.timesolver.adaptive_timestep = False
    world.timesolver.timestep = 1e-13

    def sinc_pulse(t):
        return (Bac * math.sin(2*math.pi*fmax*t)/(2*math.pi*fmax*t + 1e-30), 0.0, 0.0)
    afm.bias_magnetic_field.add_time_term(lambda t: sinc_pulse(t))

    # Zeitgrenze merken
    t_end = world.timesolver.time + runtime

    # Periodic Callback starten


    def step():
        # Lauf solange bis runtime voll ist:
        now = world.timesolver.time
        if now >= t_end:
            return  # Panel stoppt den Callback automatisch, wenn er nichts mehr tut

        # genau sample_every Zeit simulieren

        dt = min(sample_every, t_end - now)
        if dt <= 0.0:
            return
        world.timesolver.run(dt)  # <-- direkt run(dt), wie gewünscht

        # messen & streamen
        t_ns = world.timesolver.time * 1e9
        M = afm.full_magnetization.average(); n = afm.neel_vector.average()
        Mx = float(M[0]); nx = float(n[0])
    try:
        while (world.timesolver.time < t_end):
            step()
    except:
        pass
    finally:
        print(mon.max_utilization())