# Planetary Thermal Simulation Summary

This document summarizes key formulas, constants, and code snippets for modeling heat transfer in a planetary simulation with atmosphere, lithosphere, and asthenosphere layers.

---

## 1. Stefan–Boltzmann Radiation

- **Constant**:  
  ```rust
  pub const STEFAN_BOLTZMANN_CONST: f64 = 5.670374419e-8; // W·m⁻²·K⁻⁴
  pub const SECONDS_PER_YEAR: f64 = 31_536_000.0;
  ```

- **Function** (temperature in K, area in km² → J/year):
  ```rust
  pub fn radiated_joules_per_year(
      temperature_k: f64,
      surface_area_km2: f64,
      emissivity: f64,
  ) -> f64 {
      let surface_area_m2 = surface_area_km2 * 1_000_000.0;
      let power_watts = emissivity
          * STEFAN_BOLTZMANN_CONST
          * surface_area_m2
          * temperature_k.powi(4);
      power_watts * SECONDS_PER_YEAR
  }
  ```

---

## 2. Conductive Energy Transfer

- **Function** (adjacent layers):
  ```rust
  pub fn conductive_energy_transfer(
      k: f64,              // W/m·K
      t_from: f64,
      t_to: f64,
      thickness_m: f64,
      dt_seconds: f64,
      area_m2: f64
  ) -> f64 {
      let flux = k * (t_from - t_to) / thickness_m;  // W/m²
      flux * dt_seconds * area_m2                   // J
  }
  ```

---

## 3. Attenuation Through Lithosphere

- **Thermal resistance**:  
  R = d / k  (m²·K/W)

- **Attenuation factor**:
  ```rust
  pub fn compute_attenuation(k: f64, thickness_m: f64, r0: f64) -> f64 {
      let resistance = thickness_m / k;
      let attenuation = 1.0 / (1.0 + (resistance / r0));
      attenuation.clamp(0.0, 1.0)
  }
  pub const R0_EARTH_LITHOSPHERE: f64 = 20_000.0; // m²·K/W
  ```

---

## 4. Core Heat Input

- **Earth estimate**:  
  ≈1.3e21 J/year total →  
  **2.52e12 J/km²/year**

  ```rust
  pub const CORE_HEAT_FLOW_J_PER_KM2_PER_YEAR: f64 = 2.52e12;
  ```

---

## 5. Direct Radiation Fraction Through Atmosphere

- **Exponential (Beer–Lambert)**:  
  τ = κ·M  
  f_direct = exp(–κ·M)

- **Logistic**:  
  f_direct = 1 / (1 + (M/M50)^n)

  ```rust
  pub const ATM_OPACITY_KAPPA: f64 = 4.6e-4; // m²/kg

  enum RadianceModel {
      Exponential { kappa: f64 },
      Logistic { m50: f64, n: f64 },
  }

  pub fn direct_radiation_fraction(
      atm_mass_m2: f64,
      model: RadianceModel
  ) -> f64 {
      match model {
          RadianceModel::Exponential { kappa } => (-kappa * atm_mass_m2).exp(),
          RadianceModel::Logistic { m50, n } => 1.0 / (1.0 + (atm_mass_m2 / m50).powf(n)),
      }
  }
  ```

---

## 6. Material Melting Range

- **Add to `Material`**:
  ```rust
  pub struct Material {
      pub kind: MaterialType,
      pub melting_point_range_k: (f64, f64), // e.g., (1215.0, 1260.0)
      // ...
  }
  ```

- **Randomize**:
  ```rust
  let (min_k, max_k) = material.melting_point_range_k;
  let melting_point = rand::thread_rng().gen_range(min_k..=max_k);
  ```

---

### Usage

Import these functions and constants into your simulation modules and apply them at each timestep (e.g., 5000-year steps) to drive realistic heat transfer, lithosphere growth/melt, and radiative exchange.

*Generated on simulation modeling best practices.*
