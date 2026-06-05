use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tracing::{info, warn, error};

use crate::error::{MisakiError, Result};
use crate::config::{PolicyConfig, ModelConfig};
use crate::providers::{ChatMessage, ModelRequest, Provider, TokenUsage};
use crate::schema::SchemaValidator;
use crate::validation::{parse_json, score_confidence, extract_evidence};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceMetadata {
    pub models_tried: Vec<String>,
    pub schema_valid: bool,
    pub business_rules_passed: bool,
    pub cost_usd: f32,
    pub latency_ms: u64,
    pub retries: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractResult<T> {
    pub value: T,
    pub confidence: HashMap<String, f32>,
    pub evidence: HashMap<String, String>,
    pub trace: TraceMetadata,
}

pub struct RecoveryEngine<'a> {
    providers: &'a HashMap<String, Arc<dyn Provider>>,
    models_config: &'a HashMap<String, ModelConfig>,
    circuit_breakers: &'a HashMap<String, Arc<crate::circuit_breaker::CircuitBreaker>>,
}

impl<'a> RecoveryEngine<'a> {
    pub fn new(
        providers: &'a HashMap<String, Arc<dyn Provider>>,
        models_config: &'a HashMap<String, ModelConfig>,
        circuit_breakers: &'a HashMap<String, Arc<crate::circuit_breaker::CircuitBreaker>>,
    ) -> Self {
        Self {
            providers,
            models_config,
            circuit_breakers,
        }
    }

    /// Resolves model alias to Provider client, model name, and provider ID
    fn resolve_model(&self, alias: &str) -> Result<(Arc<dyn Provider>, String, String)> {
        let model_cfg = self.models_config.get(alias).ok_or_else(|| {
            MisakiError::Config(format!("Model alias '{}' not found in configuration", alias))
        })?;

        let provider = self.providers.get(&model_cfg.provider).ok_or_else(|| {
            MisakiError::Config(format!(
                "Provider '{}' for model alias '{}' not found in configuration",
                model_cfg.provider, alias
            ))
        })?;

        Ok((provider.clone(), model_cfg.model.clone(), model_cfg.provider.clone()))
    }

    /// Estimates cost in USD for known API models
    fn estimate_cost(&self, model: &str, usage: &TokenUsage) -> f32 {
        let (input_cost_per_m, output_cost_per_m) = match model {
            m if m.contains("gpt-4o-mini") => (0.150, 0.600),
            m if m.contains("gpt-4o") => (2.50, 10.00),
            m if m.contains("claude-3-5-sonnet") => (3.00, 15.00),
            m if m.contains("claude-3-haiku") => (0.25, 1.25),
            _ => (0.0, 0.0), // Treat local/vllm/unrecognized models as free or self-hosted
        };

        let input_usd = (usage.prompt_tokens as f32 / 1_000_000.0) * input_cost_per_m;
        let output_usd = (usage.completion_tokens as f32 / 1_000_000.0) * output_cost_per_m;
        input_usd + output_usd
    }

    pub async fn execute_cascade(
        &self,
        policy: &PolicyConfig,
        source_text: &str,
        schema: &SchemaValidator,
        system_instructions: Option<&str>,
    ) -> Result<ExtractResult<Value>> {
        let system_prompt = format!(
            "{}\n\nYou must return valid JSON that conforms strictly to this JSON Schema:\n{}",
            system_instructions.unwrap_or("Extract structured information from the input text."),
            serde_json::to_string_pretty(schema.raw_schema()).unwrap_or_default()
        );

        let messages = vec![
            ChatMessage {
                role: "system".to_string(),
                content: system_prompt,
            },
            ChatMessage {
                role: "user".to_string(),
                content: format!("Document:\n\n{}", source_text),
            },
        ];

        self.execute_cascade_internal(policy, messages, schema, Some(source_text.to_string())).await
    }

    pub async fn execute_cascade_messages(
        &self,
        policy: &PolicyConfig,
        mut messages: Vec<ChatMessage>,
        schema: &SchemaValidator,
    ) -> Result<ExtractResult<Value>> {
        let schema_instructions = format!(
            "\n\nYou must return valid JSON that conforms strictly to this JSON Schema:\n{}",
            serde_json::to_string_pretty(schema.raw_schema()).unwrap_or_default()
        );

        let mut found_system = false;
        for msg in &mut messages {
            if msg.role == "system" {
                msg.content.push_str(&schema_instructions);
                found_system = true;
                break;
            }
        }

        if !found_system {
            messages.insert(0, ChatMessage {
                role: "system".to_string(),
                content: format!("Extract structured information.{}", schema_instructions),
            });
        }

        // Find the last user message to use as the source text for evidence matching
        let source_text = messages.iter()
            .rfind(|m| m.role == "user")
            .map(|m| m.content.clone());

        self.execute_cascade_internal(policy, messages, schema, source_text).await
    }

    async fn execute_cascade_internal(
        &self,
        policy: &PolicyConfig,
        initial_messages: Vec<ChatMessage>,
        schema: &SchemaValidator,
        source_text: Option<String>,
    ) -> Result<ExtractResult<Value>> {
        let start_time = Instant::now();
        let max_attempts = policy.max_attempts.unwrap_or(3);
        
        let mut models_tried = Vec::new();
        let mut total_cost_usd = 0.0;
        let mut total_calls = 0;
        let mut last_error: Option<String> = None;

        for (step_idx, step) in policy.steps.iter().enumerate() {
            info!("Executing cascade step {}: model alias '{}'", step_idx + 1, step.model);
            
            let (provider, model_name, provider_id) = match self.resolve_model(&step.model) {
                Ok(res) => res,
                Err(e) => {
                    error!("Failed to resolve model alias '{}': {}", step.model, e);
                    last_error = Some(e.to_string());
                    continue;
                }
            };

            // Check circuit breaker
            if self.circuit_breakers.get(&provider_id).is_some_and(|cb| cb.is_open()) {
                warn!("Circuit breaker for provider '{}' is OPEN. Skipping step.", provider_id);
                last_error = Some(format!("Circuit breaker for provider '{}' is open", provider_id));
                continue;
            }

            let mut messages = initial_messages.clone();

            let mut attempts = 0;
            let step_max_attempts = if step_idx == policy.steps.len() - 1 {
                max_attempts.saturating_sub(total_calls)
            } else {
                2
            };

            while attempts < step_max_attempts {
                attempts += 1;
                total_calls += 1;
                
                models_tried.push(step.model.clone());
                info!("Calling model '{}' (attempt {} of {})", model_name, attempts, step_max_attempts);

                let req = ModelRequest {
                    model: model_name.clone(),
                    messages: messages.clone(),
                    temperature: Some(0.0),
                    max_tokens: None,
                    response_schema: Some(schema.raw_schema().clone()),
                    logprobs: Some(true),
                    top_logprobs: Some(1),
                };

                let completion_future = provider.completion(&req);
                let completion_res = if let Some(ms) = step.timeout_ms {
                    match tokio::time::timeout(std::time::Duration::from_millis(ms), completion_future).await {
                        Ok(res) => res,
                        Err(_) => {
                            warn!("Timeout of {}ms exceeded calling model '{}'", ms, model_name);
                            Err(MisakiError::Provider {
                                provider: provider_id.clone(),
                                message: format!("Timeout of {}ms exceeded", ms),
                            })
                        }
                    }
                } else {
                    completion_future.await
                };

                let response = match completion_res {
                    Ok(resp) => {
                        if let Some(cb) = self.circuit_breakers.get(&provider_id) {
                            cb.record_success();
                        }
                        resp
                    }
                    Err(e) => {
                        warn!("Provider completion error: {}", e);
                        last_error = Some(e.to_string());
                        if let Some(cb) = self.circuit_breakers.get(&provider_id) {
                            cb.record_failure();
                        }
                        continue;
                    }
                };

                if let Some(ref usage) = response.token_usage {
                    let cost = self.estimate_cost(&model_name, usage);
                    total_cost_usd += cost;
                    info!("Step cost: ${:.6} (tokens: in={}, out={})", cost, usage.prompt_tokens, usage.completion_tokens);
                }

                let parsed_val = match parse_json(&response.content) {
                    Ok(json) => json,
                    Err(e) => {
                        warn!("JSON parse failed on response: {}", e);
                        last_error = Some(e.to_string());
                        
                        messages.push(ChatMessage {
                            role: "assistant".to_string(),
                            content: response.content.clone(),
                        });
                        messages.push(ChatMessage {
                            role: "user".to_string(),
                            content: format!(
                                "Failed to parse JSON. Error: {}.\nPlease output only valid JSON matching the schema.",
                                e
                            ),
                        });
                        continue;
                    }
                };

                match schema.validate(&parsed_val) {
                    Ok(()) => {
                        let confidence = score_confidence(&parsed_val, attempts, attempts > 1, true, response.logprobs.as_deref());
                        
                        let accepted = if let Some(ref criteria) = step.accept_if {
                            let mut meets = true;
                            if criteria.schema_valid == Some(true) && schema.validate(&parsed_val).is_err() {
                                meets = false;
                            }
                            if let Some(req_conf) = criteria.confidence_gte {
                                for (field, score) in &confidence {
                                    if *score < req_conf {
                                        info!("Rejecting extraction: field '{}' confidence {:.2} < required {:.2}", field, score, req_conf);
                                        meets = false;
                                        break;
                                    }
                                }
                            }
                            meets
                        } else {
                            true
                        };

                        if accepted || step_idx == policy.steps.len() - 1 {
                            info!("Accepting extracted object. Cascade successful.");
                            let evidence = extract_evidence(&parsed_val, source_text.as_deref().unwrap_or(""));
                            
                            let trace = TraceMetadata {
                                models_tried,
                                schema_valid: true,
                                business_rules_passed: true,
                                cost_usd: total_cost_usd,
                                latency_ms: start_time.elapsed().as_millis() as u64,
                                retries: total_calls.saturating_sub(1),
                            };

                            return Ok(ExtractResult {
                                value: parsed_val,
                                confidence,
                                evidence,
                                trace,
                            });
                        } else {
                            info!("Confidence criteria not met. Escalating to next step.");
                            break;
                        }
                    }
                    Err(e) => {
                        warn!("Schema validation failed: {}", e);
                        last_error = Some(e.to_string());

                        messages.push(ChatMessage {
                            role: "assistant".to_string(),
                            content: response.content.clone(),
                        });
                        messages.push(ChatMessage {
                            role: "user".to_string(),
                            content: format!(
                                "The JSON failed schema validation.\nErrors: {}\n\nPlease output the corrected JSON matching the schema.",
                                e
                            ),
                        });
                    }
                }
            }
        }

        let final_err = last_error.unwrap_or_else(|| {
            "All cascade steps failed without a clean error".to_string()
        });

        error!("Cascade execution exhausted all options. Final error: {}", final_err);
        Err(MisakiError::RecoveryFailed {
            attempts: total_calls,
            last_error: final_err,
        })
    }
}
