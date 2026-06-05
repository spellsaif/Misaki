use thiserror::Error;

#[derive(Error, Debug)]
pub enum MisakiError {
    #[error("Schema compilation error: {0}")]
    SchemaCompilation(String),

    #[error("Validation error: {0}")]
    Validation(String),

    #[error("Provider error from {provider}: {message}")]
    Provider {
        provider: String,
        message: String,
    },

    #[error("Request failed after {attempts} attempts. Last error: {last_error}")]
    RecoveryFailed {
        attempts: usize,
        last_error: String,
    },

    #[error("JSON serialization/deserialization error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("HTTP client error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Unknown error: {0}")]
    Unknown(String),
}

pub type Result<T> = std::result::Result<T, MisakiError>;
