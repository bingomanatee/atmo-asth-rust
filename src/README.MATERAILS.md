# Materials JSON Reference

This README explains each property defined in the `materials.json` table. Every material has three phase entries—`solid`, `liquid`, and `gas`—each with the following fields:

* **density\_kg\_m3** (kg·m⁻³)

    * Mass per unit volume for the material in that phase.
    * Used to convert between mass and volume in thermal and mechanical calculations.

* **specific\_heat\_capacity\_j\_per\_kg\_k** (J·kg⁻¹·K⁻¹)

    * Energy required to raise 1 kg of the material by 1 K at constant pressure in that phase.
    * Governs how much heat must be added or removed to change the temperature.

* **thermal\_conductivity\_w\_m\_k** (W·m⁻¹·K⁻¹)

    * Rate at which heat flows through a 1 m thick slab of the material with a 1 K temperature difference.
    * Used in Fourier’s law to model conduction.

* **thermal\_transmission\_r0\_min** (minutes)

    * Minimum characteristic time for thermal transmission (R₀) in this phase.
    * Defines the fastest timescale for heat exchange with neighboring cells or environment.

* **thermal\_transmission\_r0\_max** (minutes)

    * Maximum characteristic time for thermal transmission (R₀) in this phase.
    * Defines the slowest timescale for heat exchange; useful for stochastic or range-based models.

* **melt\_temp** (K)

    * The melting (solidus) temperature at which the material begins phase change from solid to liquid.
    * Heat added beyond sensible heating is diverted into the latent fusion bank at this temperature.

* **latent\_heat\_fusion** (J·kg⁻¹)

    * Energy required to convert 1 kg of the material from solid to liquid at `melt_temp`, without changing temperature.
    * Calculated as $Q = m 	imes L_{\mathrm{fusion}}$.

* **boil\_temp** (K)

    * The boiling (vaporization) temperature at which the material begins phase change from liquid to gas.
    * Heat added beyond sensible heating of the liquid is diverted into the vaporization bank at this temperature.

* **latent\_heat\_vapor** (J·kg⁻¹)

    * Energy required to convert 1 kg of the material from liquid to gas at `boil_temp`, without changing temperature.
    * Calculated as $Q = m 	imes L_{\mathrm{vaporization}}$.

---

## Usage

1. **Load the JSON** at runtime with your preferred JSON parser (e.g., Serde in Rust).
2. **Look up** each material by its key (e.g., "granite").
3. **Select** the appropriate phase entry based on current state.
4. **Use** these properties in your energy, diffusion, and phase-change calculations.
