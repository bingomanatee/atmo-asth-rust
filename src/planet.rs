use h3o::Resolution;

pub struct Planet {
    pub radius_km: f64,
    pub resolution: Resolution,
}

impl Planet {
    pub fn new(radius: f64, res: Resolution) -> Planet {
        Planet {
            radius_km: radius,
            resolution: res,
        }
    }
}
