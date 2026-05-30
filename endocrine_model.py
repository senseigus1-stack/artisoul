# endocrine_model.py
# Advanced endocrine system simulator with non‑linear kinetics,
# circadian rhythms, transport delays, and menstrual cycle dynamics.
# All variables are scaled between 0 and 1 (1 = upper normal limit).

import numpy as np

class AdvancedEndocrineSystem:
    def __init__(self, sex='female', baseline_anxiety=0.7, baseline_social=0.2, baseline_energy=0.6):
        self.sex = sex
        self.baseline_anxiety = baseline_anxiety
        self.baseline_social = baseline_social
        self.baseline_energy = baseline_energy

        # Major hormones and neurotransmitters
        self.cortisol = 0.3
        self.serotonin = 0.6
        self.dopamine = 0.5
        self.noradrenaline = 0.2
        self.oxytocin = 0.3
        self.t3 = 0.5                     # triiodothyronine (thyroid)

        # Gonadal axis
        self.gnrh = 0.4                   # gonadotropin-releasing hormone
        self.lh = 0.4                     # luteinizing hormone
        self.testosterone = 0.6 if sex == 'male' else 0.1
        self.estradiol = 0.1 if sex == 'male' else 0.5
        self.progesterone = 0.2 if sex == 'female' else 0.0

        # HPA axis intermediates
        self.crh = 0.3                    # corticotropin-releasing hormone
        self.acth = 0.3                   # adrenocorticotropic hormone

        # Metabolism
        self.glucose = 0.6
        self.insulin = 0.5

        # Adaptive receptors
        self.gr_sensitivity = 0.8         # glucocorticoid receptor sensitivity
        self.ht1a_sensitivity = 0.5       # 5-HT1A autoreceptor sensitivity

        # Allostatic load (cumulative wear and tear)
        self.allostatic_load = 0.2

        # Transport delays (lagged copies)
        self.crh_delayed = self.crh
        self.acth_delayed = self.acth

        # Internal simulation time (minutes)
        self.sim_time = 0.0

        # Menstrual phase (0..1, 0 = first day of menstruation)
        self.menstrual_phase = 0.0

    # -------------------------------------------------------------------------
    # Hill functions (sigmoidal activation / inhibition)
    # -------------------------------------------------------------------------
    def hill_activation(self, x, K, n):
        """Activation: x^n / (K^n + x^n)"""
        return x**n / (K**n + x**n)

    def hill_inhibition(self, x, K, n):
        """Inhibition: K^n / (K^n + x^n)"""
        return K**n / (K**n + x**n)

    # -------------------------------------------------------------------------
    # Access to current hormone values as a dictionary
    # -------------------------------------------------------------------------
    def get_hormone_dict(self):
        return {
            'cortisol': self.cortisol,
            'serotonin': self.serotonin,
            'dopamine': self.dopamine,
            'noradrenaline': self.noradrenaline,
            'oxytocin': self.oxytocin,
            't3': self.t3,
            'testosterone': self.testosterone if self.sex == 'male' else self.estradiol,
            'allostatic_load': self.allostatic_load,
            'menstrual_phase': self.menstrual_phase
        }

    # -------------------------------------------------------------------------
    # Human‑readable description of the internal state
    # -------------------------------------------------------------------------
    def state_description(self):
        c, s, d = self.cortisol, self.serotonin, self.dopamine
        na, oxy, t3 = self.noradrenaline, self.oxytocin, self.t3
        t = self.testosterone if self.sex == 'male' else self.estradiol
        allo = self.allostatic_load

        anxiety = (c * 0.7 + na * 0.3 + (1 - s) * 0.2) * 100
        depression = ((1 - s) * 0.6 + (1 - d) * 0.4) * 100
        irritability = max(0, (c - s) * 100 + na * 40)
        energy = (t3 * 0.5 + self.glucose * 0.3 + d * 0.2) * 100
        social_interest = oxy * 100
        burnout = allo * 100
        libido = t * 100

        # Menstrual phase description
        ph = self.menstrual_phase
        if ph < 0.14:
            cycle_desc = "menstruation, feeling low"
        elif ph < 0.28:
            cycle_desc = "follicular phase, energy rising"
        elif ph < 0.42:
            cycle_desc = "ovulation, peak attractiveness and libido"
        elif ph < 0.57:
            cycle_desc = "luteal phase, PMS starting"
        elif ph < 0.71:
            cycle_desc = "PMS, irritability and emotional instability"
        else:
            cycle_desc = "premenstrual dip, everything is annoying"

        if allo > 0.6:
            chronic = "chronic exhaustion, apathy"
        elif allo > 0.3:
            chronic = "accumulated fatigue, reduced motivation"
        else:
            chronic = "relative freshness"

        if anxiety > 70:
            anx = "panicky alertness"
        elif anxiety > 40:
            anx = "elevated anxiety"
        else:
            anx = "calm"

        if depression > 70:
            mood = "deep sadness"
        elif depression > 40:
            mood = "low mood"
        else:
            mood = "stable mood"

        return (f"[cycle phase: {cycle_desc}; {anx} ({anxiety:.0f}%), {mood}, "
                f"irritability {irritability:.0f}%, energy {energy:.0f}%, "
                f"social interest {social_interest:.0f}%, burnout {burnout:.0f}%, "
                f"libido {libido:.0f}%, {chronic}]")

    # -------------------------------------------------------------------------
    # ODE right‑hand side: computes derivatives of all state variables
    # -------------------------------------------------------------------------
    def _derivatives(self, state, stress, comfort, dt_min):
        """
        dt_min: time since last call in minutes (used for circadian phase).
        Returns list of derivatives in the same order as the state vector.
        """
        (crh, acth, cort, na, sero, da, oxy, t3,
         gnrh, lh, test, est, prog, gluc, ins,
         allo, gr_sens, ht1a_sens, crh_del, acth_del) = state

        # Circadian rhythms (1440 minutes per day)
        ph = (self.sim_time / 1440) % 1.0
        self.menstrual_phase = (ph * 24 / 28) % 1.0   # 28‑day cycle
        menstrual_phase = self.menstrual_phase

        cort_rhythm = 0.3 + 0.25 * np.sin(2*np.pi*(ph - 0.33)) + 0.1 * np.sin(4*np.pi*(ph - 0.33))
        sero_rhythm = 0.5 + 0.2 * np.sin(2*np.pi*(ph - 0.5))
        t3_rhythm   = 0.5 + 0.1 * np.sin(2*np.pi*(ph - 0.25))
        t_rhythm    = 0.05 * np.sin(2*np.pi*(ph - 0.3))

        # PMS factor: amplifies negative mood during late luteal phase
        if 0.5 <= menstrual_phase < 0.9:
            pms_factor = 1.5 + 0.5 * np.sin(2*np.pi*(menstrual_phase - 0.5)/0.4)
        else:
            pms_factor = 1.0

        # --- HPA axis (non‑linear, with delays) ---
        crh_stim = 0.4 * stress + 0.15 * allo
        crh_inhib = self.hill_inhibition(cort, 0.5, 3) * gr_sens
        d_crh = crh_stim * crh_inhib - 0.2 * crh
        d_crh_del = (crh - crh_del) / 5.0          # ~5 min delay

        acth_stim = 0.6 * crh_del
        acth_inhib = self.hill_inhibition(cort, 0.5, 2) * gr_sens
        d_acth = acth_stim * acth_inhib - 0.25 * acth
        d_acth_del = (acth - acth_del) / 8.0       # ~8 min delay

        d_cort = 0.5 * acth_del + 0.2 * cort_rhythm - 0.1 * cort

        # --- Noradrenaline (fast stress response) ---
        d_na = 0.4 * stress - 0.2 * na

        # --- Serotonin (mood; suppressed by PMS) ---
        sero_synthesis = (0.1 * comfort + 0.05 * sero_rhythm - 0.05 * stress * (1 + allo)) / pms_factor
        d_ht1a = (0.02 * sero - 0.01 * ht1a_sens) * 0.1
        sero_release = sero_synthesis * (1 - 0.5 * ht1a_sens)
        d_sero = sero_release - 0.1 * sero

        # --- Dopamine (reward & novelty) ---
        novelty = abs(stress - comfort)
        reward = comfort * 0.8 + 0.2 * novelty
        d_da = 0.1 * reward + 0.05 * novelty - 0.05 * cort - 0.15 * da

        # --- Oxytocin (social bonding) ---
        oxy_synthesis = comfort * 0.25 * self.baseline_social
        d_oxy = oxy_synthesis - 0.02 * oxy - 0.1 * cort * oxy

        # --- Thyroid axis (T3) ---
        t3_stim = 0.01 * comfort - 0.02 * stress + 0.02 * t3_rhythm
        d_t3 = t3_stim + 0.01 * (self.baseline_energy - t3)

        # --- Gonadal axis ---
        gnrh_stim = 0.2 * (1 - cort) - 0.05 * allo
        d_gnrh = gnrh_stim - 0.02 * gnrh
        lh_stim = 0.4 * gnrh
        if self.sex == 'female':
            # LH surge around ovulation (phase ~0.5)
            lh_surge = 0.5 * np.exp(-50 * (menstrual_phase - 0.5)**2)
            lh_stim += lh_surge
        d_lh = lh_stim - 0.03 * lh

        if self.sex == 'male':
            d_test = 0.1 * lh + 0.02 * t_rhythm - 0.05 * cort - 0.01 * test
            d_est = 0.0
            d_prog = 0.0
        else:
            d_test = 0.0
            d_est = 0.1 * lh - 0.03 * cort - 0.01 * est
            prog_stim = 0.1 * lh * (1 if menstrual_phase > 0.5 else 0.2)
            d_prog = prog_stim - 0.02 * prog

        # --- Metabolism (glucose / insulin) ---
        d_gluc = 0.1 * na + 0.05 * stress - 0.05 * gluc * (1 + 0.5 * t3)
        d_ins = 0.1 * comfort - 0.03 * ins

        # --- Allostatic load ---
        if cort > 0.6:
            allo_inc = 0.02 * (cort - 0.5)
        else:
            allo_inc = -0.01
        d_allo = allo_inc

        # --- Receptor dynamics ---
        if cort > 0.5:
            d_gr = -0.01
        else:
            d_gr = 0.005

        return [d_crh, d_acth, d_cort, d_na, d_sero, d_da, d_oxy, d_t3,
                d_gnrh, d_lh, d_test, d_est, d_prog, d_gluc, d_ins,
                d_allo, d_gr, d_ht1a, d_crh_del, d_acth_del]

    # -------------------------------------------------------------------------
    # Integration step (Euler with sub‑steps, dt in minutes)
    # -------------------------------------------------------------------------
    def step(self, dt, stress, comfort):
        """Advance the simulation by dt minutes using constant stress/comfort."""
        if dt <= 0:
            return
        substep = 0.1                 # 0.1 minute for stability
        n_steps = max(1, int(dt / substep))
        actual_dt = dt / n_steps

        for _ in range(n_steps):
            state = [self.crh, self.acth, self.cortisol,
                     self.noradrenaline, self.serotonin, self.dopamine,
                     self.oxytocin, self.t3,
                     self.gnrh, self.lh,
                     self.testosterone, self.estradiol, self.progesterone,
                     self.glucose, self.insulin,
                     self.allostatic_load, self.gr_sensitivity, self.ht1a_sensitivity,
                     self.crh_delayed, self.acth_delayed]

            d_state = self._derivatives(state, stress, comfort, actual_dt)

            # Update variables (Euler)
            self.crh += d_state[0] * actual_dt
            self.acth += d_state[1] * actual_dt
            self.cortisol += d_state[2] * actual_dt
            self.noradrenaline += d_state[3] * actual_dt
            self.serotonin += d_state[4] * actual_dt
            self.dopamine += d_state[5] * actual_dt
            self.oxytocin += d_state[6] * actual_dt
            self.t3 += d_state[7] * actual_dt
            self.gnrh += d_state[8] * actual_dt
            self.lh += d_state[9] * actual_dt
            self.testosterone += d_state[10] * actual_dt
            self.estradiol += d_state[11] * actual_dt
            self.progesterone += d_state[12] * actual_dt
            self.glucose += d_state[13] * actual_dt
            self.insulin += d_state[14] * actual_dt
            self.allostatic_load += d_state[15] * actual_dt
            self.gr_sensitivity += d_state[16] * actual_dt
            self.ht1a_sensitivity += d_state[17] * actual_dt
            self.crh_delayed += d_state[18] * actual_dt
            self.acth_delayed += d_state[19] * actual_dt

            # Clamp all values to [0, 1]
            for attr in ['crh', 'acth', 'cortisol', 'noradrenaline', 'serotonin',
                         'dopamine', 'oxytocin', 't3', 'gnrh', 'lh',
                         'testosterone', 'estradiol', 'progesterone',
                         'glucose', 'insulin', 'allostatic_load',
                         'gr_sensitivity', 'ht1a_sensitivity', 'crh_delayed', 'acth_delayed']:
                val = getattr(self, attr)
                setattr(self, attr, max(0.0, min(1.0, val)))

            self.sim_time += actual_dt