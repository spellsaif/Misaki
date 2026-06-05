use async_trait::async_trait;
use std::sync::Mutex;
use crate::error::Result;
use crate::providers::{ModelRequest, ModelResponse, Provider, TokenUsage};

pub struct MockProvider {
    pub call_count: Mutex<usize>,
    pub responses: Vec<String>,
    pub sleep_duration: Option<std::time::Duration>,
    pub logprobs: Option<Vec<crate::providers::LogprobInfo>>,
}

impl MockProvider {
    pub fn new(responses: Vec<String>) -> Self {
        Self {
            call_count: Mutex::new(0),
            responses,
            sleep_duration: None,
            logprobs: None,
        }
    }

    pub fn with_sleep(mut self, duration: std::time::Duration) -> Self {
        self.sleep_duration = Some(duration);
        self
    }

    pub fn with_logprobs(mut self, logprobs: Vec<crate::providers::LogprobInfo>) -> Self {
        self.logprobs = Some(logprobs);
        self
    }
}

#[async_trait]
impl Provider for MockProvider {
    fn name(&self) -> &str {
        "mock"
    }

    async fn completion(&self, _request: &ModelRequest) -> Result<ModelResponse> {
        if let Some(dur) = self.sleep_duration {
            tokio::time::sleep(dur).await;
        }

        let mut count = self.call_count.lock().unwrap();
        let idx = *count;
        *count += 1;

        let content = if idx < self.responses.len() {
            self.responses[idx].clone()
        } else {
            "{}".to_string()
        };

        Ok(ModelResponse {
            content,
            token_usage: Some(TokenUsage {
                prompt_tokens: 10,
                completion_tokens: 20,
                total_tokens: 30,
            }),
            logprobs: self.logprobs.clone(),
        })
    }
}
