use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use crate::error::{MisakiError, Result};
use crate::providers::{ModelRequest, ModelResponse, Provider, TokenUsage};

pub struct GeminiProvider {
    client: Client,
    base_url: String,
    api_key: Option<String>,
}

impl GeminiProvider {
    pub fn new(base_url: String, api_key: Option<String>) -> Self {
        Self {
            client: Client::new(),
            base_url,
            api_key,
        }
    }
}

#[derive(Serialize)]
struct GeminiGenerateRequest {
    contents: Vec<GeminiContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system_instruction: Option<GeminiInstruction>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "generationConfig")]
    generation_config: Option<GeminiConfig>,
}

#[derive(Serialize)]
struct GeminiContent {
    role: String,
    parts: Vec<GeminiPart>,
}

#[derive(Serialize)]
struct GeminiPart {
    text: String,
}

#[derive(Serialize)]
struct GeminiInstruction {
    parts: Vec<GeminiPart>,
}

#[derive(Serialize)]
struct GeminiConfig {
    #[serde(skip_serializing_if = "Option::is_none", rename = "responseMimeType")]
    response_mime_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "responseSchema")]
    response_schema: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
}

#[derive(Deserialize)]
struct GeminiResponse {
    candidates: Option<Vec<GeminiCandidate>>,
    #[serde(rename = "usageMetadata")]
    usage_metadata: Option<GeminiUsage>,
}

#[derive(Deserialize)]
struct GeminiCandidate {
    content: GeminiResponseContent,
}

#[derive(Deserialize)]
struct GeminiResponseContent {
    parts: Vec<GeminiResponsePart>,
}

#[derive(Deserialize)]
struct GeminiResponsePart {
    text: Option<String>,
}

#[derive(Deserialize)]
struct GeminiUsage {
    #[serde(rename = "promptTokenCount")]
    prompt_token_count: u32,
    #[serde(rename = "candidatesTokenCount")]
    candidates_token_count: u32,
}

#[async_trait]
impl Provider for GeminiProvider {
    fn name(&self) -> &str {
        "gemini"
    }

    async fn completion(&self, request: &ModelRequest) -> Result<ModelResponse> {
        // Build url. Gemini API URL includes the model name in the path.
        // Format: https://generativelanguage.googleapis.com/v1beta/models/<model>:generateContent
        let url = format!(
            "{}/v1beta/models/{}:generateContent",
            self.base_url.trim_end_matches('/'),
            request.model
        );

        // Translate system messages out of contents array
        let mut system_contents = Vec::new();
        let mut contents = Vec::new();

        for msg in &request.messages {
            if msg.role == "system" {
                system_contents.push(msg.content.clone());
            } else {
                // Map roles: user -> user, assistant -> model
                let role = match msg.role.as_str() {
                    "assistant" => "model".to_string(),
                    _ => "user".to_string(),
                };
                contents.push(GeminiContent {
                    role,
                    parts: vec![GeminiPart {
                        text: msg.content.clone(),
                    }],
                });
            }
        }

        let system_instruction = if system_contents.is_empty() {
            None
        } else {
            Some(GeminiInstruction {
                parts: vec![GeminiPart {
                    text: system_contents.join("\n\n"),
                }],
            })
        };

        // If a schema is requested, configure generationConfig to require JSON schema
        let generation_config = if request.response_schema.is_some() || request.temperature.is_some() {
            let (mime, schema) = if let Some(ref s) = request.response_schema {
                (Some("application/json".to_string()), Some(s.clone()))
            } else {
                (None, None)
            };
            Some(GeminiConfig {
                response_mime_type: mime,
                response_schema: schema,
                temperature: request.temperature,
            })
        } else {
            None
        };

        let api_req = GeminiGenerateRequest {
            contents,
            system_instruction,
            generation_config,
        };

        let mut req_builder = self.client.post(&url)
            .header("content-type", "application/json");

        // Use x-goog-api-key header for authentication
        if let Some(ref key) = self.api_key {
            req_builder = req_builder.header("x-goog-api-key", key);
        }

        let http_res = req_builder
            .json(&api_req)
            .send()
            .await?;

        if !http_res.status().is_success() {
            let status = http_res.status();
            let err_body = http_res.text().await.unwrap_or_default();
            return Err(MisakiError::Provider {
                provider: "gemini".to_string(),
                message: format!("HTTP error {}: {}", status, err_body),
            });
        }

        let api_res: GeminiResponse = http_res.json().await?;
        
        let candidates = api_res.candidates.ok_or_else(|| {
            MisakiError::Provider {
                provider: "gemini".to_string(),
                message: "No candidates returned in response".to_string(),
            }
        })?;

        let first_candidate = candidates.first().ok_or_else(|| {
            MisakiError::Provider {
                provider: "gemini".to_string(),
                message: "Candidates list was empty".to_string(),
            }
        })?;

        let mut content = String::new();
        for part in &first_candidate.content.parts {
            if let Some(ref text) = part.text {
                content.push_str(text);
            }
        }

        let token_usage = api_res.usage_metadata.map(|u| TokenUsage {
            prompt_tokens: u.prompt_token_count,
            completion_tokens: u.candidates_token_count,
            total_tokens: u.prompt_token_count + u.candidates_token_count,
        });

        Ok(ModelResponse { content, token_usage, logprobs: None })
    }
}
