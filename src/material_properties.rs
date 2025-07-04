use crate::material_composite::{MaterialCompositeType, MaterialPhase};
use crate::material_parser::MaterialParser;
use std::path::{Path, PathBuf};
use once_cell::sync::Lazy;
use serde_json::Value;

/// Default path to the materials JSON file
static DEFAULT_MATERIALS_PATH: Lazy<PathBuf> = Lazy::new(|| {
    PathBuf::from("src/materials.json")
});

/// Material properties utility for simulation use
pub struct MaterialProperties;

impl MaterialProperties {
    /// Get the default materials file path
    pub fn default_file_path() -> &'static Path {
        &DEFAULT_MATERIALS_PATH
    }
    
    /// Reload the materials cache
    pub fn reload_cache() {
        MaterialParser::clear_cache();
    }
    
    /// Get a specific property for a material type and phase
    pub fn get_property<P: AsRef<Path>>(
        file_path: P,
        material_type: MaterialCompositeType,
        phase: MaterialPhase,
        property: &str,
    ) -> Result<f64, String> {
        let material_name = match material_type {
            MaterialCompositeType::Silicate => "silicate",
            MaterialCompositeType::Basaltic => "basalt",
            MaterialCompositeType::Granitic => "granite",
            MaterialCompositeType::Metallic => "steel",
            MaterialCompositeType::Icy => "water",
            MaterialCompositeType::Air => "air",
        };
        
        let phase_name = match phase {
            MaterialPhase::Solid => "solid",
            MaterialPhase::Liquid => "liquid",
            MaterialPhase::Gas => "gas",
        };
        
        let json = MaterialParser::load_json(file_path)?;
        
        Self::extract_property(&json, material_name, phase_name, property)
    }
    
    /// Extract a property from JSON data
    fn extract_property(
        json: &Value,
        material_name: &str,
        phase_name: &str,
        property: &str,
    ) -> Result<f64, String> {
        json.get(material_name)
            .and_then(|m| m.get(phase_name))
            .and_then(|p| p.get(property))
            .and_then(|v| v.as_f64())
            .ok_or_else(|| format!("Property '{}' not found for {}/{}", property, material_name, phase_name))
    }
    
    /// Get a property using the default materials file
    pub fn get_default_property(
        material_type: MaterialCompositeType,
        phase: MaterialPhase,
        property: &str,
    ) -> Result<f64, String> {
        Self::get_property(Self::default_file_path(), material_type, phase, property)
    }
    
    /// Get a property from an embedded JSON string
    pub fn get_property_from_str(
        key: &str,
        json_str: &str,
        material_type: MaterialCompositeType,
        phase: MaterialPhase,
        property: &str,
    ) -> Result<f64, String> {
        let material_name = match material_type {
            MaterialCompositeType::Silicate => "silicate",
            MaterialCompositeType::Basaltic => "basalt",
            MaterialCompositeType::Granitic => "granite",
            MaterialCompositeType::Metallic => "steel",
            MaterialCompositeType::Icy => "water",
            MaterialCompositeType::Air => "air",
        };
        
        let phase_name = match phase {
            MaterialPhase::Solid => "solid",
            MaterialPhase::Liquid => "liquid",
            MaterialPhase::Gas => "gas",
        };
        
        let json = MaterialParser::load_json_str(key, json_str)?;
        
        Self::extract_property(&json, material_name, phase_name, property)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_get_property_from_str() {
        // Test JSON string
        let json_str = r#"{
            "silicate": {
                "solid": {
                    "density_kg_m3": 3300.0,
                    "specific_heat_capacity_j_per_kg_k": 1200.0
                }
            }
        }"#;
        
        // Test getting properties from the string
        let density = MaterialProperties::get_property_from_str(
            "test_json",
            json_str,
            MaterialCompositeType::Silicate,
            MaterialPhase::Solid,
            "density_kg_m3"
        ).unwrap();
        
        assert_eq!(density, 3300.0);
        
        let specific_heat = MaterialProperties::get_property_from_str(
            "test_json",
            json_str,
            MaterialCompositeType::Silicate,
            MaterialPhase::Solid,
            "specific_heat_capacity_j_per_kg_k"
        ).unwrap();
        
        assert_eq!(specific_heat, 1200.0);
    }
    
    #[test]
    fn test_error_handling() {
        // Test JSON string with missing property
        let json_str = r#"{
            "silicate": {
                "solid": {
                    "density_kg_m3": 3300.0
                }
            }
        }"#;
        
        // Test non-existent property
        let result = MaterialProperties::get_property_from_str(
            "test_json",
            json_str,
            MaterialCompositeType::Silicate,
            MaterialPhase::Solid,
            "non_existent_property"
        );
        
        assert!(result.is_err());
        
        // Test non-existent material
        let result = MaterialProperties::get_property_from_str(
            "test_json",
            json_str,
            MaterialCompositeType::Metallic,
            MaterialPhase::Solid,
            "density_kg_m3"
        );
        
        assert!(result.is_err());
        
        // Test non-existent phase
        let result = MaterialProperties::get_property_from_str(
            "test_json",
            json_str,
            MaterialCompositeType::Silicate,
            MaterialPhase::Gas,
            "density_kg_m3"
        );
        
        assert!(result.is_err());
    }
}