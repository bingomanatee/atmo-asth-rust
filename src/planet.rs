use h3o::Resolution;

#[derive(Debug, Clone)]
pub struct Planet {
    pub radius_km: f64,
    pub resolution: Resolution,
    pub mass_kg: f64,
    pub gravity_m_s2: f64,
    pub name: String,
}

impl Planet {
    pub fn new(radius: f64, res: Resolution) -> Planet {
        // Calculate Earth-like mass and gravity from radius
        let earth_radius_km = 6371.0;
        let earth_mass_kg = 5.972e24;
        let earth_gravity = 9.81;

        // Scale mass by volume ratio (assuming similar density)
        let volume_ratio = (radius / earth_radius_km).powi(3);
        let mass_kg = earth_mass_kg * volume_ratio;

        // Calculate gravity: g = GM/r²
        let gravity_m_s2 = (6.674e-11 * mass_kg) / ((radius * 1000.0).powi(2));

        Planet {
            radius_km: radius,
            resolution: res,
            mass_kg,
            gravity_m_s2,
            name: "Planet".to_string(),
        }
    }

    /// Create Earth with realistic properties
    pub fn earth(res: Resolution) -> Planet {
        Planet {
            radius_km: 6371.0,
            resolution: res,
            mass_kg: 5.972e24,
            gravity_m_s2: 9.81,
            name: "Earth".to_string(),
        }
    }

    /// Create Mars with realistic properties
    pub fn mars(res: Resolution) -> Planet {
        Planet {
            radius_km: 3390.0,
            resolution: res,
            mass_kg: 6.39e23,
            gravity_m_s2: 3.71,
            name: "Mars".to_string(),
        }
    }

    /// Create Venus with realistic properties
    pub fn venus(res: Resolution) -> Planet {
        Planet {
            radius_km: 6052.0,
            resolution: res,
            mass_kg: 4.867e24,
            gravity_m_s2: 8.87,
            name: "Venus".to_string(),
        }
    }

    /// Create custom planet with specified properties
    pub fn custom(name: &str, radius_km: f64, mass_kg: f64, res: Resolution) -> Planet {
        // Calculate gravity from mass and radius
        let gravity_m_s2 = (6.674e-11 * mass_kg) / ((radius_km * 1000.0).powi(2));

        Planet {
            radius_km,
            resolution: res,
            mass_kg,
            gravity_m_s2,
            name: name.to_string(),
        }
    }
}
