use std::collections::HashMap;
use std::sync::Arc;
use serde::{Deserialize, Serialize};
use schemars::JsonSchema;
use misaki_core::{Misaki, ExtractResult};
use misaki_core::config::{Config, ProviderConfig, ModelConfig, PolicyConfig, PolicyStep, AcceptCriteria};
use misaki_core::providers::mock::MockProvider;

#[derive(Serialize, Deserialize, JsonSchema, Debug, Clone, PartialEq)]
struct MockInvoice {
    invoice_number: String,
    amount: f64,
}

#[tokio::test]
async fn test_cascade_and_repair() {
    // 1. Create a config with two mock models in a cascade policy
    let mut providers = HashMap::new();
    providers.insert(
        "mock_prov".to_string(),
        ProviderConfig {
            provider_type: "mock".to_string(),
            api_key_env: None,
            base_url: None,
        },
    );

    let mut models = HashMap::new();
    models.insert(
        "cheap-mock".to_string(),
        ModelConfig {
            provider: "mock_prov".to_string(),
            model: "cheap-model".to_string(),
        },
    );
    models.insert(
        "strong-mock".to_string(),
        ModelConfig {
            provider: "mock_prov".to_string(),
            model: "strong-model".to_string(),
        },
    );

    let mut policies = HashMap::new();
    policies.insert(
        "cheap-then-strong".to_string(),
        PolicyConfig {
            strategy: "cascade".to_string(),
            steps: vec![
                PolicyStep {
                    model: "cheap-mock".to_string(),
                    accept_if: Some(AcceptCriteria {
                        schema_valid: Some(true),
                        confidence_gte: Some(0.95),
                    }),
                    timeout_ms: None,
                },
                PolicyStep {
                    model: "strong-mock".to_string(),
                    accept_if: None,
                    timeout_ms: None,
                },
            ],
            max_cost_usd: None,
            max_attempts: Some(4),
        },
    );

    let config = Config {
        server: None,
        providers,
        models,
        policies,
    };

    let mut misaki = Misaki::new(config).unwrap();

    // 2. Set up mock responses.
    // - Response 1 (cheap-mock, attempt 1): Malformed JSON (triggers repair prompt)
    // - Response 2 (cheap-mock, attempt 2): Valid JSON but wrong types (amount is string, schema validation fails, cascades to strong-mock)
    // - Response 3 (strong-mock, attempt 1): Correct invoice JSON!
    let mock_responses = vec![
        "{ malformed json".to_string(),
        r#"{"invoice_number": "INV-123", "amount": "one hundred"}"#.to_string(),
        r#"{"invoice_number": "INV-123", "amount": 100.0}"#.to_string(),
    ];

    let mock_provider = Arc::new(MockProvider::new(mock_responses));
    misaki.add_provider("mock_prov".to_string(), mock_provider.clone());

    // 3. Execute extraction
    let result: ExtractResult<MockInvoice> = misaki
        .extract::<MockInvoice>()
        .from("This is an invoice INV-123 for 100.0 dollars")
        .policy("cheap-then-strong")
        .await
        .unwrap();

    assert_eq!(result.value.invoice_number, "INV-123");
    assert_eq!(result.value.amount, 100.0);
    assert!(result.trace.schema_valid);
    assert_eq!(result.trace.models_tried.len(), 3); // cheap-mock (attempt 1), cheap-mock (attempt 2), strong-mock (attempt 1)
    assert_eq!(result.trace.retries, 2);
    assert_eq!(*mock_provider.call_count.lock().unwrap(), 3);
    
    // Check evidence matches
    assert!(result.evidence.contains_key("invoice_number"));
    assert!(result.evidence.contains_key("amount"));
    assert!(result.evidence.get("invoice_number").unwrap().contains("invoice INV-123"));
}

#[tokio::test]
async fn test_timeouts() {
    let mut providers = HashMap::new();
    providers.insert(
        "slow_prov".to_string(),
        ProviderConfig {
            provider_type: "mock".to_string(),
            api_key_env: None,
            base_url: None,
        },
    );
    providers.insert(
        "fast_prov".to_string(),
        ProviderConfig {
            provider_type: "mock".to_string(),
            api_key_env: None,
            base_url: None,
        },
    );

    let mut models = HashMap::new();
    models.insert(
        "slow-model".to_string(),
        ModelConfig {
            provider: "slow_prov".to_string(),
            model: "slow-model-name".to_string(),
        },
    );
    models.insert(
        "fast-model".to_string(),
        ModelConfig {
            provider: "fast_prov".to_string(),
            model: "fast-model-name".to_string(),
        },
    );

    let mut policies = HashMap::new();
    policies.insert(
        "timeout-cascade".to_string(),
        PolicyConfig {
            strategy: "cascade".to_string(),
            steps: vec![
                PolicyStep {
                    model: "slow-model".to_string(),
                    accept_if: None,
                    timeout_ms: Some(20), // 20ms timeout limit
                },
                PolicyStep {
                    model: "fast-model".to_string(),
                    accept_if: None,
                    timeout_ms: None,
                },
            ],
            max_cost_usd: None,
            max_attempts: Some(3),
        },
    );

    let config = Config {
        server: None,
        providers,
        models,
        policies,
    };

    let mut misaki = Misaki::new(config).unwrap();

    // slow_prov sleeps for 100ms (will timeout)
    let slow_provider = Arc::new(
        MockProvider::new(vec![r#"{"invoice_number": "TIMEOUT", "amount": 0.0}"#.to_string()])
            .with_sleep(std::time::Duration::from_millis(100))
    );
    // fast_prov returns instantly
    let fast_provider = Arc::new(
        MockProvider::new(vec![r#"{"invoice_number": "FAST", "amount": 99.0}"#.to_string()])
    );

    misaki.add_provider("slow_prov".to_string(), slow_provider.clone());
    misaki.add_provider("fast_prov".to_string(), fast_provider.clone());

    let result: ExtractResult<MockInvoice> = misaki
        .extract::<MockInvoice>()
        .from("Doc")
        .policy("timeout-cascade")
        .await
        .unwrap();

    // Fast provider should have succeeded because slow provider timed out
    assert_eq!(result.value.invoice_number, "FAST");
    assert_eq!(result.value.amount, 99.0);
    assert_eq!(result.trace.models_tried, vec!["slow-model", "slow-model", "fast-model"]);
}

#[tokio::test]
async fn test_circuit_breakers() {
    let mut providers = HashMap::new();
    providers.insert(
        "flaky_prov".to_string(),
        ProviderConfig {
            provider_type: "mock".to_string(),
            api_key_env: None,
            base_url: None,
        },
    );
    providers.insert(
        "backup_prov".to_string(),
        ProviderConfig {
            provider_type: "mock".to_string(),
            api_key_env: None,
            base_url: None,
        },
    );

    let mut models = HashMap::new();
    models.insert(
        "flaky-model".to_string(),
        ModelConfig {
            provider: "flaky_prov".to_string(),
            model: "flaky-model-name".to_string(),
        },
    );
    models.insert(
        "backup-model".to_string(),
        ModelConfig {
            provider: "backup_prov".to_string(),
            model: "backup-model-name".to_string(),
        },
    );

    let mut policies = HashMap::new();
    policies.insert(
        "circuit-breaker-cascade".to_string(),
        PolicyConfig {
            strategy: "cascade".to_string(),
            steps: vec![
                PolicyStep {
                    model: "flaky-model".to_string(),
                    accept_if: None,
                    timeout_ms: None,
                },
                PolicyStep {
                    model: "backup-model".to_string(),
                    accept_if: None,
                    timeout_ms: None,
                },
            ],
            max_cost_usd: None,
            max_attempts: Some(2),
        },
    );

    let config = Config {
        server: None,
        providers,
        models,
        policies,
    };

    let mut misaki = Misaki::new(config).unwrap();

    // 1. Manually trip the circuit breaker for flaky_prov by calling record_failure 3 times
    let cb = misaki.circuit_breakers.get("flaky_prov").unwrap();
    cb.record_failure();
    cb.record_failure();
    cb.record_failure();
    assert!(cb.is_open());

    // 2. Set up providers. Flaky returns correct value but shouldn't be called because circuit is open!
    let flaky_provider = Arc::new(MockProvider::new(vec![r#"{"invoice_number": "SHOULD_NOT_CALL", "amount": 0.0}"#.to_string()]));
    let backup_provider = Arc::new(MockProvider::new(vec![r#"{"invoice_number": "BACKUP", "amount": 5.0}"#.to_string()]));

    misaki.add_provider("flaky_prov".to_string(), flaky_provider.clone());
    misaki.add_provider("backup_prov".to_string(), backup_provider.clone());

    // 3. Execute cascade extraction
    let result: ExtractResult<MockInvoice> = misaki
        .extract::<MockInvoice>()
        .from("Doc")
        .policy("circuit-breaker-cascade")
        .await
        .unwrap();

    // Backup provider should have been selected because flaky's circuit was open
    assert_eq!(result.value.invoice_number, "BACKUP");
    assert_eq!(*flaky_provider.call_count.lock().unwrap(), 0); // Verify flaky provider was never called!
    assert_eq!(*backup_provider.call_count.lock().unwrap(), 1);
}

#[tokio::test]
async fn test_cache() {
    let mut providers = HashMap::new();
    providers.insert(
        "cached_prov".to_string(),
        ProviderConfig {
            provider_type: "mock".to_string(),
            api_key_env: None,
            base_url: None,
        },
    );

    let mut models = HashMap::new();
    models.insert(
        "cache-model".to_string(),
        ModelConfig {
            provider: "cached_prov".to_string(),
            model: "cache-model-name".to_string(),
        },
    );

    let mut policies = HashMap::new();
    policies.insert(
        "cache-policy".to_string(),
        PolicyConfig {
            strategy: "cascade".to_string(),
            steps: vec![
                PolicyStep {
                    model: "cache-model".to_string(),
                    accept_if: None,
                    timeout_ms: None,
                },
            ],
            max_cost_usd: None,
            max_attempts: Some(1),
        },
    );

    let config = Config {
        server: None,
        providers,
        models,
        policies,
    };

    let mut misaki = Misaki::new(config).unwrap();

    let mock_provider = Arc::new(MockProvider::new(vec![r#"{"invoice_number": "CACHED_VAL", "amount": 42.0}"#.to_string()]));
    misaki.add_provider("cached_prov".to_string(), mock_provider.clone());

    // Call 1: should be a cache miss, calling the provider
    let result1: ExtractResult<MockInvoice> = misaki
        .extract::<MockInvoice>()
        .from("Doc contents")
        .policy("cache-policy")
        .await
        .unwrap();

    assert_eq!(result1.value.invoice_number, "CACHED_VAL");
    assert_eq!(*mock_provider.call_count.lock().unwrap(), 1);
    assert_eq!(result1.trace.models_tried, vec!["cache-model"]);

    // Call 2: identical request, should be a cache hit
    let result2: ExtractResult<MockInvoice> = misaki
        .extract::<MockInvoice>()
        .from("Doc contents")
        .policy("cache-policy")
        .await
        .unwrap();

    assert_eq!(result2.value.invoice_number, "CACHED_VAL");
    assert_eq!(*mock_provider.call_count.lock().unwrap(), 1); // call count remains 1!
    assert_eq!(result2.trace.models_tried, vec!["cache"]); // marked as cache hit!
}

#[tokio::test]
async fn test_logprobs_confidence() {
    use misaki_core::providers::LogprobInfo;

    let mut providers = HashMap::new();
    providers.insert(
        "logprob_prov".to_string(),
        ProviderConfig {
            provider_type: "mock".to_string(),
            api_key_env: None,
            base_url: None,
        },
    );

    let mut models = HashMap::new();
    models.insert(
        "logprob-model".to_string(),
        ModelConfig {
            provider: "logprob_prov".to_string(),
            model: "logprob-model-name".to_string(),
        },
    );

    let mut policies = HashMap::new();
    policies.insert(
        "logprob-policy".to_string(),
        PolicyConfig {
            strategy: "cascade".to_string(),
            steps: vec![
                PolicyStep {
                    model: "logprob-model".to_string(),
                    accept_if: None,
                    timeout_ms: None,
                },
            ],
            max_cost_usd: None,
            max_attempts: Some(1),
        },
    );

    let config = Config {
        server: None,
        providers,
        models,
        policies,
    };

    let mut misaki = Misaki::new(config).unwrap();

    // Setup mock logprobs:
    // logprob 1: -0.1 -> e^-0.1 = ~0.9048
    // logprob 2: -0.2 -> e^-0.2 = ~0.8187
    // Avg prob: (0.9048 + 0.8187) / 2 = ~0.8618
    // Base confidence for 1 attempt: 0.98
    // Blended: (0.98 + 0.8618) / 2 = ~0.9209
    let logprobs = vec![
        LogprobInfo { token: "{".to_string(), logprob: -0.1 },
        LogprobInfo { token: "}".to_string(), logprob: -0.2 },
    ];

    let mock_provider = Arc::new(
        MockProvider::new(vec![r#"{"invoice_number": "INV-LOG", "amount": 123.4}"#.to_string()])
            .with_logprobs(logprobs)
    );
    misaki.add_provider("logprob_prov".to_string(), mock_provider.clone());

    let result: ExtractResult<MockInvoice> = misaki
        .extract::<MockInvoice>()
        .from("Doc with logprobs")
        .policy("logprob-policy")
        .await
        .unwrap();

    assert_eq!(result.value.invoice_number, "INV-LOG");
    assert!(result.confidence.contains_key("invoice_number"));
    assert!(result.confidence.contains_key("amount"));

    let score = result.confidence.get("invoice_number").unwrap();
    // Score should be close to 0.9209
    assert!((score - 0.9209).abs() < 0.001, "Expected score around 0.9209, got {}", score);
}
