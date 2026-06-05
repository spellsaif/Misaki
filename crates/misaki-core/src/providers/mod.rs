use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use crate::error::Result;

pub mod openai;
pub mod mock;
pub mod anthropic;
pub mod gemini;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
    pub response_schema: Option<Value>,
    pub logprobs: Option<bool>,
    pub top_logprobs: Option<u32>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LogprobInfo {
    pub token: String,
    pub logprob: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelResponse {
    pub content: String,
    pub token_usage: Option<TokenUsage>,
    pub logprobs: Option<Vec<LogprobInfo>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TokenUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[async_trait]
pub trait Provider: Send + Sync {
    /// Returns the provider identifier (e.g. "openai", "anthropic")
    fn name(&self) -> &str;

    /// Sends a request to the provider to get a structured or unstructured completion
    async fn completion(&self, request: &ModelRequest) -> Result<ModelResponse>;
}
