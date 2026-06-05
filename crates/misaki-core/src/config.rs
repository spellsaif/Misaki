use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Config {
    pub server: Option<ServerConfig>,
    pub providers: HashMap<String, ProviderConfig>,
    pub models: HashMap<String, ModelConfig>,
    pub policies: HashMap<String, PolicyConfig>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProviderConfig {
    #[serde(rename = "type")]
    pub provider_type: String,
    pub api_key_env: Option<String>,
    pub base_url: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelConfig {
    pub provider: String,
    pub model: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PolicyConfig {
    pub strategy: String, // e.g., "cascade"
    pub steps: Vec<PolicyStep>,
    pub max_cost_usd: Option<f32>,
    pub max_attempts: Option<usize>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PolicyStep {
    pub model: String,
    pub accept_if: Option<AcceptCriteria>,
    pub timeout_ms: Option<u64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AcceptCriteria {
    pub schema_valid: Option<bool>,
    pub confidence_gte: Option<f32>,
}
