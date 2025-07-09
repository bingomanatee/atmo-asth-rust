## Practical Summary of Atmospheric Density vs. Altitude

### 1. Scale Height

```text
scaleHeight = (gasConstant * temperature) / (molarMass * gravity)
```

- **gasConstant** = 8.314 J/(mol·K)  
- **temperature** = absolute temperature in Kelvin (e.g. 288 K at sea level)  
- **molarMass** = mean molar mass of the air in kg/mol (≈ 0.028964 kg/mol for dry Earth air)  
- **gravity** = gravitational acceleration in m/s² (≈ 9.81 m/s² on Earth)  

**Example (Earth):**

```text
scaleHeight ≈ (8.314 × 288) / (0.028964 × 9.81) ≈ 8 400 m
```

---

### 2. Density at Altitude \(z\)

```text
densityAtAltitude(z) = densityAtSeaLevel * exp( - z / scaleHeight )
```

- **densityAtSeaLevel** = ≈ 1.225 kg/m³ (at z = 0 m)  
- **z** = height above the reference level in meters  

Every “scaleHeight” (≈ 8.4 km), density falls by a factor of e (~2.718).

---

### 3. Pressure at Altitude \(z\)

```text
pressureAtAltitude(z) = pressureAtSeaLevel * exp( - z / scaleHeight )
```

- **pressureAtSeaLevel** = ≈ 101 325 Pa (standard sea‐level pressure)

---

### 4. Mass of Air Column Above Altitude \(z\)

```text
columnMassAbove(z) = pressureAtAltitude(z) / gravity
```

- Gives the total mass per square meter of atmosphere above height z.  
- **At sea level (z = 0):**

  ```text
  columnMassAbove(0) = 101 325 Pa / 9.81 m/s² ≈ 10 333 kg/m²
  ```

- **At one scale height up (z ≈ 8 400 m):**

  ```text
  columnMassAbove(8400) ≈ 10 333 kg/m² / e ≈ 3 800 kg/m²
  ```

---

### 5. Applying to Any Planet

1. Plug in your planet’s **gravity**.  
2. Use the universal **gasConstant** and your atmosphere’s **molarMass**.  
3. If **temperature** varies with altitude, compute a “local” scale height at each layer or integrate the full barometric formula.
