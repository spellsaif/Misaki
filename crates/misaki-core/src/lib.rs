use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use serde::de::DeserializeOwned;
use serde_json::Value;

pub mod config;
pub mod error;
pub mod providers;
pub mod schema;
pub mod validation;
pub mod recovery;
pub mod circuit_breaker;
pub mod cache;

pub use error::{MisakiError, Result};
pub use recovery::{ExtractResult, TraceMetadata};
pub use schema::SchemaValidator;

use crate::config::Config;
use crate::providers::{Provider, openai::OpenAIProvider};
use crate::recovery::RecoveryEngine;

pub struct Misaki {
    config: Config,
    providers: HashMap<String, Arc<dyn Provider>>,
    pub circuit_breakers: HashMap<String, Arc<circuit_breaker::CircuitBreaker>>,
    pub cache: Arc<cache::ExactCache>,
}

impl Misaki {
    pub fn new(config: Config) -> Result<Self> {
        let mut providers = HashMap::new();
        let mut circuit_breakers = HashMap::new();

        for name in config.providers.keys() {
            circuit_breakers.insert(
                name.clone(),
                Arc::new(circuit_breaker::CircuitBreaker::new(3, std::time::Duration::from_secs(30))),
            );
        }

        for (name, prov_cfg) in &config.providers {
            let api_key = if let Some(ref env_var) = prov_cfg.api_key_env {
                std::env::var(env_var).ok()
            } else {
                None
            };

            let provider: Arc<dyn Provider> = match prov_cfg.provider_type.as_str() {
                "openai" => {
                    let base_url = prov_cfg.base_url.clone().unwrap_or_else(|| "https://api.openai.com/v1".to_string());
                    Arc::new(OpenAIProvider::new(base_url, api_key))
                }
                "openai_compatible" => {
                    let base_url = prov_cfg.base_url.clone().ok_or_else(|| {
                        MisakiError::Config(format!("base_url is required for provider '{}' of type openai_compatible", name))
                    })?;
                    Arc::new(OpenAIProvider::new(base_url, api_key))
                }
                "anthropic" => {
                    let base_url = prov_cfg.base_url.clone().unwrap_or_else(|| "https://api.anthropic.com".to_string());
                    Arc::new(providers::anthropic::AnthropicProvider::new(base_url, api_key))
                }
                "gemini" => {
                    let base_url = prov_cfg.base_url.clone().unwrap_or_else(|| "https://generativelanguage.googleapis.com".to_string());
                    Arc::new(providers::gemini::GeminiProvider::new(base_url, api_key))
                }
                "mock" => {
                    Arc::new(providers::mock::MockProvider::new(vec!["{}".to_string()]))
                }
                t => {
                    return Err(MisakiError::Config(format!("Unsupported provider type '{}' for provider '{}'", t, name)));
                }
            };

            providers.insert(name.clone(), provider);
        }

        let cache = Arc::new(cache::ExactCache::new(std::time::Duration::from_secs(300)));
        Ok(Self { config, providers, circuit_breakers, cache })
    }

    /// Programmatically registers a provider (e.g., for testing with mock clients)
    pub fn add_provider(&mut self, name: String, provider: Arc<dyn Provider>) {
        self.providers.insert(name, provider);
    }

    /// Exposes the configuration loaded by the engine
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Factory method for typed extraction requests using schemars
    pub fn extract<T>(&self) -> ExtractBuilder<'_, T>
    where
        T: schemars::JsonSchema + DeserializeOwned + Send,
    {
        ExtractBuilder::new(self)
    }

    /// Raw dynamic extraction taking schema as serde_json::Value
    pub fn extract_raw(&self) -> RawExtractBuilder<'_> {
        RawExtractBuilder::new(self)
    }
}

pub struct ExtractBuilder<'a, T> {
    misaki: &'a Misaki,
    input: Option<String>,
    policy_name: Option<String>,
    system_instructions: Option<String>,
    _phantom: std::marker::PhantomData<T>,
}

impl<'a, T> ExtractBuilder<'a, T>
where
    T: schemars::JsonSchema + DeserializeOwned + Send,
{
    pub fn new(misaki: &'a Misaki) -> Self {
        Self {
            misaki,
            input: None,
            policy_name: None,
            system_instructions: None,
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn from(mut self, input: impl Into<String>) -> Self {
        self.input = Some(input.into());
        self
    }

    pub fn policy(mut self, policy: impl Into<String>) -> Self {
        self.policy_name = Some(policy.into());
        self
    }

    pub fn system_instructions(mut self, instructions: impl Into<String>) -> Self {
        self.system_instructions = Some(instructions.into());
        self
    }

    pub async fn execute(self) -> Result<ExtractResult<T>> {
        // Generate JSON schema from T using draft 07 compatible with jsonschema crate
        let mut settings = schemars::r#gen::SchemaSettings::draft07();
        settings.inline_subschemas = true;
        let schema_generator = settings.into_generator();
        let schema_root = schema_generator.into_root_schema_for::<T>();
        let schema_val = serde_json::to_value(schema_root)?;

        let raw_result = self.misaki.extract_raw()
            .schema(schema_val)
            .from(self.input.ok_or_else(|| MisakiError::Validation("No input document provided".to_string()))?)
            .policy(self.policy_name.ok_or_else(|| MisakiError::Validation("No extraction policy specified".to_string()))?)
            .system_instructions(self.system_instructions.unwrap_or_default())
            .execute()
            .await?;

        let parsed_value: T = serde_json::from_value(raw_result.value)?;

        Ok(ExtractResult {
            value: parsed_value,
            confidence: raw_result.confidence,
            evidence: raw_result.evidence,
            trace: raw_result.trace,
        })
    }
}

// Implement IntoFuture to allow direct `.await` on the builder
impl<'a, T> IntoFuture for ExtractBuilder<'a, T>
where
    T: schemars::JsonSchema + DeserializeOwned + Send + 'static,
{
    type Output = Result<ExtractResult<T>>;
    type IntoFuture = Pin<Box<dyn Future<Output = Self::Output> + Send + 'a>>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(self.execute())
    }
}

pub struct RawExtractBuilder<'a> {
    misaki: &'a Misaki,
    schema: Option<Value>,
    input: Option<String>,
    messages: Option<Vec<crate::providers::ChatMessage>>,
    policy_name: Option<String>,
    system_instructions: Option<String>,
}

impl<'a> RawExtractBuilder<'a> {
    pub fn new(misaki: &'a Misaki) -> Self {
        Self {
            misaki,
            schema: None,
            input: None,
            messages: None,
            policy_name: None,
            system_instructions: None,
        }
    }

    pub fn schema(mut self, schema: Value) -> Self {
        self.schema = Some(schema);
        self
    }

    pub fn from(mut self, input: impl Into<String>) -> Self {
        self.input = Some(input.into());
        self
    }

    pub fn messages(mut self, messages: Vec<crate::providers::ChatMessage>) -> Self {
        self.messages = Some(messages);
        self
    }

    pub fn policy(mut self, policy: impl Into<String>) -> Self {
        self.policy_name = Some(policy.into());
        self
    }

    pub fn system_instructions(mut self, instructions: impl Into<String>) -> Self {
        self.system_instructions = Some(instructions.into());
        self
    }

    pub async fn execute(self) -> Result<ExtractResult<Value>> {
        let schema_val = self.schema.ok_or_else(|| {
            MisakiError::Validation("JSON Schema must be specified for dynamic extraction".to_string())
        })?;
        let policy_name = self.policy_name.ok_or_else(|| {
            MisakiError::Validation("Policy must be specified".to_string())
        })?;

        // 1. Serialize request inputs to build cache key
        let schema_json = serde_json::to_string(&schema_val).unwrap_or_default();
        let messages_json = if let Some(ref msgs) = self.messages {
            serde_json::to_string(msgs).unwrap_or_default()
        } else {
            let input_text = self.input.as_deref().unwrap_or("");
            let inst = self.system_instructions.as_deref().unwrap_or("");
            format!("input: {} | inst: {}", input_text, inst)
        };

        // 2. Query cache
        if let Some(mut cached_res) = self.misaki.cache.get(&messages_json, &schema_json) {
            tracing::info!("Cache hit! Returning cached result.");
            cached_res.trace.models_tried = vec!["cache".to_string()];
            return Ok(cached_res);
        }

        let schema_validator = SchemaValidator::new(schema_val)?;

        let policy = self.misaki.config.policies.get(&policy_name).ok_or_else(|| {
            MisakiError::Config(format!("Extraction policy '{}' not found in configuration", policy_name))
        })?;

        let engine = RecoveryEngine::new(&self.misaki.providers, &self.misaki.config.models, &self.misaki.circuit_breakers);
        
        let result = if let Some(msgs) = self.messages {
            engine.execute_cascade_messages(policy, msgs, &schema_validator).await?
        } else {
            let input_text = self.input.ok_or_else(|| {
                MisakiError::Validation("Input document or message history must be specified".to_string())
            })?;
            let instructions = if self.system_instructions.as_deref().unwrap_or("").is_empty() {
                None
            } else {
                self.system_instructions.as_deref()
            };

            engine.execute_cascade(policy, &input_text, &schema_validator, instructions).await?
        };

        // 3. Cache successful result
        self.misaki.cache.set(&messages_json, &schema_json, result.clone(), None);

        Ok(result)
    }
}

// Implement IntoFuture to allow direct `.await` on the raw builder
impl<'a> IntoFuture for RawExtractBuilder<'a> {
    type Output = Result<ExtractResult<Value>>;
    type IntoFuture = Pin<Box<dyn Future<Output = Self::Output> + Send + 'a>>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(self.execute())
    }
}
