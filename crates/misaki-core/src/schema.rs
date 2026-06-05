use serde_json::Value;
use crate::error::{MisakiError, Result};

#[derive(Clone, Debug)]
pub struct SchemaValidator {
    raw_schema: Value,
}

impl SchemaValidator {
    pub fn new(schema: Value) -> Result<Self> {
        // Test compilation to ensure schema is valid
        if let Err(e) = jsonschema::validator_for(&schema) {
            return Err(MisakiError::SchemaCompilation(e.to_string()));
        }
        Ok(Self { raw_schema: schema })
    }

    pub fn validate(&self, value: &Value) -> Result<()> {
        let validator = jsonschema::validator_for(&self.raw_schema)
            .map_err(|e| MisakiError::SchemaCompilation(e.to_string()))?;
            
        if let Err(e) = validator.validate(value) {
            let path = e.instance_path.to_string();
            let msg = if path.is_empty() {
                format!("Root: {}", e)
            } else {
                format!("path '{}': {}", path, e)
            };
            return Err(MisakiError::Validation(format!(
                "JSON does not match schema: {}",
                msg
            )));
        }
        Ok(())
    }

    pub fn validate_partial(&self, value: &Value) -> Result<()> {
        let validator = jsonschema::validator_for(&self.raw_schema)
            .map_err(|e| MisakiError::SchemaCompilation(e.to_string()))?;
            
        let mut first_unrecoverable = None;
        for err in validator.iter_errors(value) {
            match &err.kind {
                jsonschema::error::ValidationErrorKind::Required { .. } => {
                    // Required fields can be missing during streaming.
                }
                _ => {
                    first_unrecoverable = Some(err);
                    break;
                }
            }
        }

        if let Some(e) = first_unrecoverable {
            let path = e.instance_path.to_string();
            let msg = if path.is_empty() {
                format!("Root: {}", e)
            } else {
                format!("path '{}': {}", path, e)
            };
            return Err(MisakiError::Validation(format!(
                "JSON schema unrecoverable mismatch: {}",
                msg
            )));
        }
        Ok(())
    }

    pub fn raw_schema(&self) -> &Value {
        &self.raw_schema
    }
}
