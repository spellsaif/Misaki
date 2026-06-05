use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use crate::error::{MisakiError, Result};
use crate::providers::{ModelRequest, ModelResponse, Provider, TokenUsage};

pub struct AnthropicProvider {
    client: Client,
    base_url: String,
    api_key: Option<String>,
}

impl AnthropicProvider {
    pub fn new(base_url: String, api_key: Option<String>) -> Self {
        Self {
            client: Client::new(),
            base_url,
            api_key,
        }
    }
}

#[derive(Serialize)]
struct AnthropicChatRequest<'a> {
    model: &'a str,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    messages: Vec<AnthropicMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<AnthropicTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<AnthropicToolChoice>,
}

#[derive(Serialize)]
struct AnthropicMessage {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct AnthropicTool {
    name: String,
    description: String,
    input_schema: Value,
}

#[derive(Serialize)]
struct AnthropicToolChoice {
    #[serde(rename = "type")]
    choice_type: String,
    name: String,
}

#[derive(Deserialize)]
struct AnthropicChatResponse {
    content: Vec<AnthropicContentBlock>,
    usage: Option<AnthropicUsage>,
}

#[derive(Deserialize)]
#[serde(tag = "type")]
enum AnthropicContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[allow(dead_code)]
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
}

#[derive(Deserialize)]
struct AnthropicUsage {
    input_tokens: u32,
    output_tokens: u32,
}

#[async_trait]
impl Provider for AnthropicProvider {
    fn name(&self) -> &str {
        "anthropic"
    }

    async fn completion(&self, request: &ModelRequest) -> Result<ModelResponse> {
        let url = format!("{}/v1/messages", self.base_url.trim_end_matches('/'));

        // Translate system messages out of the message array
        let mut system_contents = Vec::new();
        let mut messages = Vec::new();

        for msg in &request.messages {
            if msg.role == "system" {
                system_contents.push(msg.content.clone());
            } else {
                // Ensure only user and assistant roles go to Anthropic
                let role = match msg.role.as_str() {
                    "assistant" => "assistant".to_string(),
                    _ => "user".to_string(),
                };
                messages.push(AnthropicMessage {
                    role,
                    content: msg.content.clone(),
                });
            }
        }

        let system = if system_contents.is_empty() {
            None
        } else {
            Some(system_contents.join("\n\n"))
        };

        // If a schema is requested, supply it as a tool and force the tool choice
        let mut tools = None;
        let mut tool_choice = None;

        if let Some(ref schema) = request.response_schema {
            tools = Some(vec![AnthropicTool {
                name: "extract_schema".to_string(),
                description: "Extract structured data matching the specified schema".to_string(),
                input_schema: schema.clone(),
            }]);
            tool_choice = Some(AnthropicToolChoice {
                choice_type: "tool".to_string(),
                name: "extract_schema".to_string(),
            });
        }

        let api_req = AnthropicChatRequest {
            model: &request.model,
            max_tokens: request.max_tokens.unwrap_or(2048),
            system,
            messages,
            temperature: request.temperature,
            tools,
            tool_choice,
        };

        let mut req_builder = self.client.post(&url)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json");

        if let Some(ref key) = self.api_key {
            req_builder = req_builder.header("x-api-key", key);
        }

        let http_res = req_builder
            .json(&api_req)
            .send()
            .await?;

        if !http_res.status().is_success() {
            let status = http_res.status();
            let err_body = http_res.text().await.unwrap_or_default();
            return Err(MisakiError::Provider {
                provider: "anthropic".to_string(),
                message: format!("HTTP error {}: {}", status, err_body),
            });
        }

        let api_res: AnthropicChatResponse = http_res.json().await?;
        
        let mut content = String::new();
        for block in api_res.content {
            match block {
                AnthropicContentBlock::Text { text } => {
                    content.push_str(&text);
                }
                AnthropicContentBlock::ToolUse { input, .. } => {
                    // Extract the tool call arguments back to a JSON string representation
                    content = serde_json::to_string(&input).unwrap_or_default();
                    break;
                }
            }
        }

        let token_usage = api_res.usage.map(|u| TokenUsage {
            prompt_tokens: u.input_tokens,
            completion_tokens: u.output_tokens,
            total_tokens: u.input_tokens + u.output_tokens,
        });

        Ok(ModelResponse { content, token_usage, logprobs: None })
    }
}
