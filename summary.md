# Conductivity and Phase Change Summary

This document consolidates key recommendations from our recent discussion on heat conduction and phase change modeling in thermal simulations.

## 1. Phase Change Modeling

### Latent Heat of Fusion
- **Definition:** Energy required to change a unit mass from solid to liquid at the melting point, without temperature change.  
- **Units:** J·kg⁻¹  
- **Formula:**  
  \[
    Q_{fusion} = m \times L_f
  \]  
  where:
  - \(m\) = mass (kg)  
  - \(L_f\) = latent heat of fusion (J·kg⁻¹)

### Latent Heat of Vaporization
- **Definition:** Energy required to change a unit mass from liquid to gas at the boiling point, without temperature change.  
- **Units:** J·kg⁻¹  
- **Formula:**  
  \[
    Q_{vapor} = m \times L_v
  \]  
  where:
  - \(L_v\) = latent heat of vaporization (J·kg⁻¹)

### Single-Threshold Transition
- **Approach:**  
  1. Sensible heating up to `melt_temp`;  
  2. Clamp temperature at `melt_temp` and send all additional energy into latent-heat bank up to `m * latent_heat_fusion`;  
  3. Swap phase to liquid and resume sensible heating with `cp_liquid`;  
  4. Repeat analogously at `boil_temp` for vaporization.

## 2. Material JSON Properties

Each material entry includes three phases (`solid`, `liquid`, `gas`) with:
- `density_kg_m3` (kg·m⁻³)  
- `specific_heat_capacity_j_per_kg_k` (J·kg⁻¹·K⁻¹)  
- `thermal_conductivity_w_m_k` (W·m⁻¹·K⁻¹)  
- `thermal_transmission_r0_min`, `thermal_transmission_r0_max` (dimensionless r0 bounds)  
- `melt_temp` (K)  
- `latent_heat_fusion` (J·kg⁻¹)  
- `boil_temp` (K)  
- `latent_heat_vapor` (J·kg⁻¹)  
- `max_temp_k` (K) to cap sensible heating per phase

## 3. Conductivity Recommendation

### 3.1 Physical Conductance via Fourier’s Law

For adjacent cells *i* and *j*:
- \(T_i, T_j\) (K); \(k_i, k_j\) (W·m⁻¹·K⁻¹); \(d_i, d_j\) (m); contact area \(A\) (m²).

Compute total thermal resistance:
```
R_cond = d_i/(k_i * A) + d_j/(k_j * A)
G_cond = 1 / R_cond      // W/K
```

### 3.2 Energy Transfer per Time Step

Over Δt (s):
```
Q = G_cond * (T_i - T_j) * Δt   // J
```

### 3.3 Random Variability Factor (r0)

Seed each cell with:
```
r0 ∼ Uniform(r0_min, r0_max)
r0_pair = (r0_i + r0_j) / 2
g_eff = g_raw * r0_pair
```

### 3.4 Implementation Pseudocode

```ts
for each node i:
  for each neighbor j:
    let k_i = props[i.phase].thermal_conductivity_w_m_k
    let k_j = props[j.phase].thermal_conductivity_w_m_k
    let d_i = thickness[i]
    let d_j = thickness[j]
    let A = sharedFaceArea
    let ΔT = T[i] - T[j]

    // Raw conductance per area
    let g_raw = 1 / (d_i/k_i + d_j/k_j)

    // Random variability
    let r0_pair = (node[i].r0 + node[j].r0) * 0.5
    let g_eff = g_raw * r0_pair

    // Energy transfer
    let Q = g_eff * A * Δt

    energy[i] -= Q
    energy[j] += Q
```

---

**End of Summary**  
