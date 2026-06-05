use std::sync::Arc;
use axum::{
    routing::{get, post},
    Router, Json, Extension,
    http::StatusCode,
    response::{IntoResponse, sse::{Event, KeepAlive, Sse}},
};
use serde::{Deserialize, Serialize};
use tracing::{info, error, warn};
use misaki_core::Misaki;
use misaki_core::config::Config;

#[derive(Debug, Deserialize, Serialize)]
struct OpenAIRequest {
    model: String, // Resolves to Misaki policy
    messages: Vec<misaki_core::providers::ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<ResponseFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
struct ResponseFormat {
    #[serde(rename = "type")]
    format_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    json_schema: Option<JsonSchemaFormat>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
struct JsonSchemaFormat {
    name: String,
    strict: Option<bool>,
    schema: serde_json::Value,
}

#[derive(Debug, Serialize)]
struct OpenAIResponse {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<OpenAIChoice>,
    usage: OpenAIUsage,
}

#[derive(Debug, Serialize)]
struct OpenAIChoice {
    index: usize,
    message: OpenAIChoiceMessage,
    finish_reason: String,
}

#[derive(Debug, Serialize)]
struct OpenAIChoiceMessage {
    role: String,
    content: String,
}

#[derive(Debug, Serialize)]
struct OpenAIUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

#[derive(Debug, Serialize)]
struct OpenAIStreamResponse {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<OpenAIStreamChoice>,
}

#[derive(Debug, Serialize)]
struct OpenAIStreamChoice {
    index: usize,
    delta: OpenAIStreamDelta,
    #[serde(skip_serializing_if = "Option::is_none")]
    finish_reason: Option<String>,
}

#[derive(Debug, Serialize)]
struct OpenAIStreamDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
}

#[derive(Debug, Serialize)]
struct ModelListResponse {
    object: String,
    data: Vec<ModelData>,
}

#[derive(Debug, Serialize)]
struct ModelData {
    id: String,
    object: String,
    created: u64,
    owned_by: String,
}

#[tokio::main]
async fn main() {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")))
        .init();

    // Determine config path
    let config_path = std::env::var("MISAKI_CONFIG")
        .unwrap_or_else(|_| "misaki.yaml".to_string());

    info!("Loading config from: {}", config_path);
    
    let config_content = match std::fs::read_to_string(&config_path) {
        Ok(c) => c,
        Err(e) => {
            error!("Could not read config file at {}: {}. Exiting.", config_path, e);
            std::process::exit(1);
        }
    };

    let config: Config = match serde_yaml::from_str(&config_content) {
        Ok(cfg) => cfg,
        Err(e) => {
            error!("Failed to parse YAML config: {}. Exiting.", e);
            std::process::exit(1);
        }
    };

    let host = config.server.as_ref().map(|s| s.host.clone()).unwrap_or_else(|| "0.0.0.0".to_string());
    let port = config.server.as_ref().map(|s| s.port).unwrap_or(8080);

    let misaki_engine = Misaki::new(config).unwrap_or_else(|e| {
        error!("Failed to initialize Misaki engine: {}. Exiting.", e);
        std::process::exit(1);
    });

    let misaki = Arc::new(tokio::sync::RwLock::new(misaki_engine));
    let misaki_clone = misaki.clone();
    let config_path_clone = config_path.clone();

    // Spawn config file watch task for live config hot-reloading
    tokio::spawn(async move {
        info!("Starting configuration watcher on: {}", config_path_clone);
        let mut last_metadata = std::fs::metadata(&config_path_clone).ok();
        loop {
            tokio::time::sleep(std::time::Duration::from_secs(2)).await;
            if let Ok(metadata) = std::fs::metadata(&config_path_clone) {
                let modified = metadata.modified().ok();
                let last_modified = last_metadata.as_ref().and_then(|m| m.modified().ok());
                if modified != last_modified {
                    info!("Configuration file modified. Reloading...");
                    match std::fs::read_to_string(&config_path_clone) {
                        Ok(content) => {
                            match serde_yaml::from_str::<Config>(&content) {
                                Ok(new_config) => {
                                    match Misaki::new(new_config) {
                                        Ok(new_misaki) => {
                                            let mut guard = misaki_clone.write().await;
                                            *guard = new_misaki;
                                            info!("Configuration hot-reloaded successfully in-place!");
                                        }
                                        Err(e) => {
                                            error!("Failed to initialize reloaded config: {}", e);
                                        }
                                    }
                                }
                                Err(e) => {
                                    error!("Failed to parse reloaded config: {}", e);
                                }
                            }
                        }
                        Err(e) => {
                            error!("Failed to read modified configuration file: {}", e);
                        }
                    }
                    last_metadata = Some(metadata);
                }
            }
        }
    });

    // Build routes
    let app = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/models", get(list_models))
        .layer(Extension(misaki));

    let listener_addr = format!("{}:{}", host, port);
    let listener = match tokio::net::TcpListener::bind(&listener_addr).await {
        Ok(l) => l,
        Err(e) => {
            error!("Failed to bind TCP listener to {}: {}. Exiting.", listener_addr, e);
            std::process::exit(1);
        }
    };

    info!("Misaki proxy listening on http://{}", listener_addr);
    
    if let Err(e) = axum::serve(listener, app).await {
        error!("Server error: {}", e);
    }
}

async fn chat_completions(
    Extension(misaki): Extension<Arc<tokio::sync::RwLock<Misaki>>>,
    Json(payload): Json<OpenAIRequest>,
) -> impl IntoResponse {
    info!("Received chat completion request for model (policy): {}", payload.model);

    if let Some(schema_fmt) = payload.response_format.as_ref()
        .filter(|fmt| fmt.format_type == "json_schema")
        .and_then(|fmt| fmt.json_schema.as_ref())
    {
        info!("Running Misaki validation pipeline for schema: {}", schema_fmt.name);

                let misaki_guard = misaki.read().await;
                let extract_res = misaki_guard.extract_raw()
                    .schema(schema_fmt.schema.clone())
                    .messages(payload.messages.clone())
                    .policy(payload.model.clone())
                    .await;

                match extract_res {
                    Ok(res) => {
                        let now_secs = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_secs();
                        let id = format!("chatcmpl-misaki-{}", uuid::Uuid::new_v4());
                        let model_name = payload.model.clone();

                        if payload.stream.unwrap_or(false) {
                            let content_str = serde_json::to_string(&res.value).unwrap_or_default();
                            let mut chunks = Vec::new();
                            let chars = content_str.chars().collect::<Vec<char>>();
                            let mut idx = 0;
                            while idx < chars.len() {
                                let end = std::cmp::min(idx + 4, chars.len());
                                let chunk: String = chars[idx..end].iter().collect();
                                chunks.push(chunk);
                                idx = end;
                            }

                            let mut events = Vec::new();

                            // 1. Initial role event
                            let first_res = OpenAIStreamResponse {
                                id: id.clone(),
                                object: "chat.completion.chunk".to_string(),
                                created: now_secs,
                                model: model_name.clone(),
                                choices: vec![OpenAIStreamChoice {
                                    index: 0,
                                    delta: OpenAIStreamDelta {
                                        role: Some("assistant".to_string()),
                                        content: None,
                                    },
                                    finish_reason: None,
                                }],
                            };
                            if let Ok(js) = serde_json::to_string(&first_res) {
                                events.push(Ok::<Event, std::convert::Infallible>(Event::default().data(js)));
                            }

                            // 2. Content chunk events
                            for chunk in chunks {
                                let chunk_res = OpenAIStreamResponse {
                                    id: id.clone(),
                                    object: "chat.completion.chunk".to_string(),
                                    created: now_secs,
                                    model: model_name.clone(),
                                    choices: vec![OpenAIStreamChoice {
                                        index: 0,
                                        delta: OpenAIStreamDelta {
                                            role: None,
                                            content: Some(chunk),
                                        },
                                        finish_reason: None,
                                    }],
                                };
                                if let Ok(js) = serde_json::to_string(&chunk_res) {
                                    events.push(Ok(Event::default().data(js)));
                                }
                            }

                            // 3. Final stop event
                            let stop_res = OpenAIStreamResponse {
                                id: id.clone(),
                                object: "chat.completion.chunk".to_string(),
                                created: now_secs,
                                model: model_name.clone(),
                                choices: vec![OpenAIStreamChoice {
                                    index: 0,
                                    delta: OpenAIStreamDelta {
                                        role: None,
                                        content: None,
                                    },
                                    finish_reason: Some("stop".to_string()),
                                }],
                            };
                            if let Ok(js) = serde_json::to_string(&stop_res) {
                                events.push(Ok(Event::default().data(js)));
                            }

                            // 4. [DONE] indicator
                            events.push(Ok(Event::default().data("[DONE]")));

                            let stream = futures_util::stream::iter(events);
                            return Sse::new(stream)
                                .keep_alive(KeepAlive::default())
                                .into_response();
                        }

                        let response = OpenAIResponse {
                            id,
                            object: "chat.completion".to_string(),
                            created: now_secs,
                            model: payload.model,
                            choices: vec![OpenAIChoice {
                                index: 0,
                                message: OpenAIChoiceMessage {
                                    role: "assistant".to_string(),
                                    content: serde_json::to_string(&res.value).unwrap_or_default(),
                                },
                                finish_reason: "stop".to_string(),
                            }],
                            usage: OpenAIUsage {
                                prompt_tokens: 0,
                                completion_tokens: 0,
                                total_tokens: 0,
                            },
                        };

                        return (StatusCode::OK, Json(response)).into_response();
                    }
                    Err(e) => {
                        error!("Misaki pipeline error: {}", e);
                        return (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            Json(serde_json::json!({
                                "error": {
                                    "message": format!("Misaki reliability layer error: {}", e),
                                    "type": "misaki_error",
                                    "code": 500
                                }
                            })),
                        ).into_response();
                    }
                }
    }

    warn!("Request received without response_format of type 'json_schema'.");
    (
        StatusCode::BAD_REQUEST,
        Json(serde_json::json!({
            "error": {
                "message": "Misaki requires a response_format of type 'json_schema' to enforce typed reliability.",
                "type": "invalid_request_error",
                "code": 400
            }
        })),
    ).into_response()
}

async fn list_models(
    Extension(misaki): Extension<Arc<tokio::sync::RwLock<Misaki>>>,
) -> impl IntoResponse {
    let mut data = Vec::new();
    let misaki_guard = misaki.read().await;
    
    for policy_name in misaki_guard.config().policies.keys() {
        data.push(ModelData {
            id: policy_name.clone(),
            object: "model".to_string(),
            created: 1717599600,
            owned_by: "misaki".to_string(),
        });
    }

    (
        StatusCode::OK,
        Json(ModelListResponse {
            object: "list".to_string(),
            data,
        }),
    )
}
