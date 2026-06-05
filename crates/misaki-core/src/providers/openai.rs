use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use crate::error::{MisakiError, Result};
use crate::providers::{ChatMessage, ModelRequest, ModelResponse, Provider, TokenUsage};

pub struct OpenAIProvider {
    client: Client,
    base_url: String,
    api_key: Option<String>,
}

impl OpenAIProvider {
    pub fn new(base_url: String, api_key: Option<String>) -> Self {
        Self {
            client: Client::new(),
            base_url,
            api_key,
        }
    }
}

#[derive(Serialize)]
struct OpenAIChatRequest<'a> {
    model: &'a str,
    messages: &'a [ChatMessage],
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<OpenAIResponseFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    logprobs: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_logprobs: Option<u32>,
}

#[derive(Serialize)]
struct OpenAIResponseFormat {
    #[serde(rename = "type")]
    format_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    json_schema: Option<OpenAIJsonSchema>,
}

#[derive(Serialize)]
struct OpenAIJsonSchema {
    name: String,
    strict: bool,
    schema: Value,
}

#[derive(Deserialize)]
struct OpenAIChatResponse {
    choices: Vec<OpenAIChoice>,
    usage: Option<OpenAIUsage>,
}

#[derive(Deserialize)]
struct OpenAIChoice {
    message: OpenAIChoiceMessage,
    logprobs: Option<OpenAILogprobs>,
}

#[derive(Deserialize)]
struct OpenAILogprobs {
    content: Option<Vec<OpenAILogprobInfo>>,
}

#[derive(Deserialize)]
struct OpenAILogprobInfo {
    token: String,
    logprob: f32,
}

#[derive(Deserialize)]
struct OpenAIChoiceMessage {
    content: Option<String>,
}

#[derive(Deserialize)]
struct OpenAIUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

#[async_trait]
impl Provider for OpenAIProvider {
    fn name(&self) -> &str {
        "openai"
    }

    async fn completion(&self, request: &ModelRequest) -> Result<ModelResponse> {
        let url = format!("{}/chat/completions", self.base_url.trim_end_matches('/'));
        
        let response_format = request.response_schema.as_ref().map(|schema| {
            OpenAIResponseFormat {
                format_type: "json_schema".to_string(),
                json_schema: Some(OpenAIJsonSchema {
                    name: "misaki_schema".to_string(),
                    strict: true,
                    schema: schema.clone(),
                }),
            }
        });

        let api_req = OpenAIChatRequest {
            model: &request.model,
            messages: &request.messages,
            temperature: request.temperature,
            max_tokens: request.max_tokens,
            response_format,
            logprobs: request.logprobs,
            top_logprobs: request.top_logprobs,
        };

        let mut req_builder = self.client.post(&url);
        if let Some(ref key) = self.api_key {
            req_builder = req_builder.bearer_auth(key);
        }

        let http_res = req_builder
            .json(&api_req)
            .send()
            .await?;

        if !http_res.status().is_success() {
            let status = http_res.status();
            let err_body = http_res.text().await.unwrap_or_default();
            return Err(MisakiError::Provider {
                provider: "openai".to_string(),
                message: format!("HTTP error {}: {}", status, err_body),
            });
        }

        let api_res: OpenAIChatResponse = http_res.json().await?;
        let choice = api_res.choices.first().ok_or_else(|| {
            MisakiError::Provider {
                provider: "openai".to_string(),
                message: "No choices returned in response".to_string(),
            }
        })?;

        let content = choice.message.content.clone().unwrap_or_default();
        let token_usage = api_res.usage.map(|u| TokenUsage {
            prompt_tokens: u.prompt_tokens,
            completion_tokens: u.completion_tokens,
            total_tokens: u.total_tokens,
        });

        let logprobs = choice.logprobs.as_ref().and_then(|lp| {
            lp.content.as_ref().map(|c| {
                c.iter()
                    .map(|info| crate::providers::LogprobInfo {
                        token: info.token.clone(),
                        logprob: info.logprob,
                    })
                    .collect()
            })
        });

        Ok(ModelResponse { content, token_usage, logprobs })
    }
}
