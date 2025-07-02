# Generation and Absorption of Volatiles from Melting

This document summarizes the key thermal and phase-change properties of your materials and shows how to parameterize both the **generation** of gas (via melting/vaporization) and the **absorption** (trapping) of volatiles as melt migrates through lithosphere.

---

## 1. Key Parameters

| Name        | Phase   | Melt Temp (K) | Boil Temp (K) | ΔH<sub>fusion</sub> (J/kg) | ΔH<sub>vapor</sub> (J/kg) | ρ (kg/m³) | C<sub>p</sub> (J/kg·K) | k (W/m·K) |
|-------------|---------|---------------|---------------|----------------------------|---------------------------|-----------|-----------------------|-----------|
| **Basalt**  | solid   | 1473          | 2900          | 4.0×10<sup>5</sup>         | 2.0×10<sup>6</sup>        | 3000      | 840                   | 1.8       |
|             | liquid  | 1473          | 2900          | 4.0×10<sup>5</sup>         | 2.0×10<sup>6</sup>        | 2850      | 1100                  | 1.2       |
|             | gas     | 1473          | 2900          | 4.0×10<sup>5</sup>         | 2.0×10<sup>6</sup>        | 0.1       | 1200                  | 0.1       |
| **Granite** | solid   | 1215          | 3000          | 2.25×10<sup>5</sup>        | 1.75×10<sup>6</sup>       | 2700      | 790                   | 2.5       |
|             | liquid  | 1215          | 3000          | 2.25×10<sup>5</sup>        | 1.75×10<sup>6</sup>       | 2500      | 1000                  | 1.5       |
|             | gas     | 1215          | 3000          | 2.25×10<sup>5</sup>        | 1.75×10<sup>6</sup>       | 0.1       | 1100                  | 0.1       |
| **Silicate**| solid   | 1600          | 3200          | 3.0×10<sup>5</sup>         | 1.8×10<sup>6</sup>        | 2650      | 900                   | 2.0       |
|             | liquid  | 1600          | 3200          | 3.0×10<sup>5</sup>         | 1.8×10<sup>6</sup>        | 2600      | 1100                  | 1.0       |
|             | gas     | 1600          | 3200          | 3.0×10<sup>5</sup>         | 1.8×10<sup>6</sup>        | 0.1       | 1200                  | 0.1       |
| **Steel**   | solid   | 1808          | 3133          | 2.72×10<sup>5</sup>        | 6.2×10<sup>6</sup>        | 7850      | 450                   | 43.0      |
|             | liquid  | 1808          | 3133          | 2.72×10<sup>5</sup>        | 6.2×10<sup>6</sup>        | 7000      | 500                   | 35.0      |
|             | gas     | 1808          | 3133          | 2.72×10<sup>5</sup>        | 6.2×10<sup>6</sup>        | 0.1       | 600                   | 0.1       |
| **Water**   | solid   | 273.15        | 373.15        | 3.34×10<sup>5</sup>        | 2.26×10<sup>6</sup>       | 917       | 2100                  | 2.2       |
|             | liquid  | 273.15        | 373.15        | 3.34×10<sup>5</sup>        | 2.26×10<sup>6</sup>       | 1000      | 4186                  | 0.6       |
|             | gas     | 273.15        | 373.15        | 3.34×10<sup>5</sup>        | 2.26×10<sup>6</sup>       | 0.6       | 2010                  | 0.02      |

---

## 2. Generation of Volatiles

For each hex:

1. **Melt volume flux** \(F_{\rm melt}\) (m³ m⁻² yr⁻¹)
2. **Volatile concentration** \(C_{\rm vol}\) (kg volatile / m³ melt)
3. **Degassing efficiency** \(E_p\) (fraction reaching atmosphere)

\[
\Phi_{\rm gen}
= F_{\rm melt}\;\times\;C_{\rm vol}\;\times\;E_p\;\times\;A_h
\]

—where \(A_h\) is hex area (m²).

### Converting from latent heat data

If you model heating from solid at \(T_0\) to vapor at \(T_1\):

1. Heat solid to melt:  
   \(Q_1 = C_s\,(T_{\rm melt}-T_0)\)
2. Melt fusion:  
   \(Q_2 = \Delta H_{\rm fusion}\)
3. Heat liquid to boil:  
   \(Q_3 = C_\ell\,(T_{\rm boil}-T_{\rm melt})\)
4. Vaporization:  
   \(Q_4 = \Delta H_{\rm vapor}\)
5. Heat vapor to \(T_1\):  
   \(Q_5 = C_g\,(T_1-T_{\rm boil})\)

Total energy per kg:  
\[
Q_{\rm total} = Q_1 + Q_2 + Q_3 + Q_4 + Q_5
\]

You can invert this to compute mass of vapor generated given energy input.

---

## 3. Lithospheric Absorption

Empirically, a fraction \(A_{\rm lith}\) of volatiles is trapped per km of lithosphere:

\[
k_{\rm lith} \approx 0.003\;\text{(per km)}
\quad,\quad
A_{\rm lith,hex} = \min\bigl(1,\;k_{\rm lith}\times T_{\rm lith,hex}\bigr)
\]

- **Fast upwelling**: \(k\approx0.005\)
- **Global avg.**: \(k\approx0.003\)
- **Slow intrusions**: \(k\approx0.002\)

Split flux:

\[
\Phi_{\rm atm}
= \Phi_{\rm gen}\times\bigl(1 - A_{\rm lith,hex}\bigr)
\quad,\quad
\Phi_{\rm lith}
= \Phi_{\rm gen}\times A_{\rm lith,hex}
\]

---

## 4. Example Workflow

1. **Load material data** (see table above).
2. **Choose** melt flux \(F_{\rm melt}\), volatile conc. \(C_{\rm vol}\), \(E_p\).
3. **Compute** generation \(\Phi_{\rm gen}\).
4. **Compute** absorption based on local thickness \(T_{\rm lith,hex}\).
5. **Distribute** \(\Phi_{\rm atm}\) to atmosphere reservoir; \(\Phi_{\rm lith}\) to crustal store.

---

## 5. Notes & Extensions

- You can make \(k_{\rm lith}\) depth- or temperature-dependent.
- For multi-species degassing, repeat per volatile (H₂O, CO₂, SO₂…).
- If you track energy rather than mass, use each material’s latent heats and heat capacities.
- Adjust parameters for different tectonic settings (ridges vs. arcs vs. plumes).

---

*Export this file as `generation_absorption.md` for inclusion in your simulation documentation.*
