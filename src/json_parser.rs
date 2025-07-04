use serde_json::{self, Value};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use once_cell::sync::Lazy;

/// Cache for loaded JSON files to avoid repeated disk reads
static JSON_CACHE: Lazy<Mutex<HashMap<PathBuf, Value>>> = Lazy::new(|| {
    Mutex::new(HashMap::new())
});

/// Cache for embedded JSON strings
static EMBEDDED_CACHE: Lazy<Mutex<HashMap<String, Value>>> = Lazy::new(|| {
    Mutex::new(HashMap::new())
});

/// JSON parser utility for efficient access to JSON data
pub struct JsonParser;

impl JsonParser {
    /// Load JSON data from a file path, using cache if available
    pub fn load_json<P: AsRef<Path>>(file_path: P) -> Result<Value, String> {
        let path_buf = file_path.as_ref().to_path_buf();
        
        // Try to get from cache first
        {
            let cache = JSON_CACHE.lock().unwrap();
            if let Some(json) = cache.get(&path_buf) {
                return Ok(json.clone());
            }
        }
        
        // Not in cache, load from disk
        let json_str = match fs::read_to_string(&path_buf) {
            Ok(content) => content,
            Err(e) => return Err(format!("Failed to read file {}: {}", path_buf.display(), e)),
        };
        
        // Parse JSON and store in cache
        let json: Value = match serde_json::from_str(&json_str) {
            Ok(parsed) => parsed,
            Err(e) => return Err(format!("Failed to parse JSON: {}", e)),
        };
        
        // Store in cache for future use
        {
            let mut cache = JSON_CACHE.lock().unwrap();
            cache.insert(path_buf, json.clone());
        }
        
        Ok(json)
    }
    
    /// Load JSON data from an embedded string, using cache if available
    pub fn load_json_str(key: &str, json_str: &str) -> Result<Value, String> {
        // Try to get from cache first
        {
            let cache = EMBEDDED_CACHE.lock().unwrap();
            if let Some(json) = cache.get(key) {
                return Ok(json.clone());
            }
        }
        
        // Parse JSON and store in cache
        let json: Value = match serde_json::from_str(json_str) {
            Ok(parsed) => parsed,
            Err(e) => return Err(format!("Failed to parse JSON: {}", e)),
        };
        
        // Store in cache for future use
        {
            let mut cache = EMBEDDED_CACHE.lock().unwrap();
            cache.insert(key.to_string(), json.clone());
        }
        
        Ok(json)
    }
    
    /// Get a nested value from JSON using a path of keys
    pub fn get_value<'a>(json: &'a Value, path: &[&str]) -> Option<&'a Value> {
        let mut current = json;
        for &key in path {
            current = current.get(key)?;
        }
        Some(current)
    }
    
    /// Get a nested f64 value from JSON using a path of keys
    pub fn get_f64(json: &Value, path: &[&str]) -> Option<f64> {
        Self::get_value(json, path).and_then(|v| v.as_f64())
    }
    
    /// Get a nested string value from JSON using a path of keys
    pub fn get_str<'a>(json: &'a Value, path: &[&str]) -> Option<&'a str> {
        Self::get_value(json, path).and_then(|v| v.as_str())
    }
    
    /// Get a nested boolean value from JSON using a path of keys
    pub fn get_bool(json: &Value, path: &[&str]) -> Option<bool> {
        Self::get_value(json, path).and_then(|v| v.as_bool())
    }
    
    /// Get a nested integer value from JSON using a path of keys
    pub fn get_i64(json: &Value, path: &[&str]) -> Option<i64> {
        Self::get_value(json, path).and_then(|v| v.as_i64())
    }
    
    /// Clear all JSON caches
    pub fn clear_cache() {
        let mut file_cache = JSON_CACHE.lock().unwrap();
        file_cache.clear();
        
        let mut embedded_cache = EMBEDDED_CACHE.lock().unwrap();
        embedded_cache.clear();
    }
    
    /// Get the number of cached files and embedded strings
    pub fn cache_size() -> (usize, usize) {
        let file_cache = JSON_CACHE.lock().unwrap();
        let embedded_cache = EMBEDDED_CACHE.lock().unwrap();
        (file_cache.len(), embedded_cache.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    
    #[test]
    fn test_json_str_caching() {
        // Clear cache to start fresh
        JsonParser::clear_cache();
        assert_eq!(JsonParser::cache_size(), (0, 0));
        
        // Test string JSON
        let json_str = r#"{
            "name": "test",
            "value": 42,
            "nested": {
                "bool": true,
                "array": [1, 2, 3],
                "deep": {
                    "string": "hello"
                }
            }
        }"#;
        
        // First load should cache the string
        let json1 = JsonParser::load_json_str("test_key", json_str).unwrap();
        assert_eq!(JsonParser::cache_size(), (0, 1));
        assert_eq!(json1["name"].as_str().unwrap(), "test");
        assert_eq!(json1["value"].as_i64().unwrap(), 42);
        
        // Second load should use cache
        let json2 = JsonParser::load_json_str("test_key", json_str).unwrap();
        assert_eq!(JsonParser::cache_size(), (0, 1));
        assert_eq!(json2["name"].as_str().unwrap(), "test");
        assert_eq!(json2["value"].as_i64().unwrap(), 42);
        
        // Different key should add new cache entry
        let json3 = JsonParser::load_json_str("another_key", json_str).unwrap();
        assert_eq!(JsonParser::cache_size(), (0, 2));
        assert_eq!(json3["name"].as_str().unwrap(), "test");
        
        // Test path-based value getters
        let json = JsonParser::load_json_str("test_key", json_str).unwrap();
        
        // Test get_f64
        assert_eq!(JsonParser::get_f64(&json, &["value"]), Some(42.0));
        
        // Test get_str
        assert_eq!(JsonParser::get_str(&json, &["name"]), Some("test"));
        assert_eq!(JsonParser::get_str(&json, &["nested", "deep", "string"]), Some("hello"));
        
        // Test get_bool
        assert_eq!(JsonParser::get_bool(&json, &["nested", "bool"]), Some(true));
        
        // Test get_i64
        assert_eq!(JsonParser::get_i64(&json, &["value"]), Some(42));
        
        // Clear cache
        JsonParser::clear_cache();
        assert_eq!(JsonParser::cache_size(), (0, 0));
    }
    
    #[test]
    fn test_error_handling() {
        // Invalid JSON string
        let invalid_json = r#"{"broken": "json""#;
        let result = JsonParser::load_json_str("invalid", invalid_json);
        assert!(result.is_err());
        
        // Non-existent file
        let non_existent = PathBuf::from("/path/that/does/not/exist.json");
        let result = JsonParser::load_json(non_existent);
        assert!(result.is_err());
        
        // Test path-based value getters with missing paths
        let json_str = r#"{"name": "test", "value": 42}"#;
        let json = JsonParser::load_json_str("test_key", json_str).unwrap();
        
        assert_eq!(JsonParser::get_f64(&json, &["missing"]), None);
        assert_eq!(JsonParser::get_str(&json, &["missing"]), None);
        assert_eq!(JsonParser::get_bool(&json, &["missing"]), None);
        assert_eq!(JsonParser::get_i64(&json, &["missing"]), None);
        
        assert_eq!(JsonParser::get_f64(&json, &["name", "missing"]), None);
        assert_eq!(JsonParser::get_str(&json, &["value", "missing"]), None);
    }
}