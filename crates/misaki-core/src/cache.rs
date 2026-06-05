use std::collections::HashMap;
use std::sync::RwLock;
use std::time::{Duration, Instant};
use serde_json::Value;
use crate::recovery::ExtractResult;

#[derive(Clone, Debug)]
struct CacheEntry {
    result: ExtractResult<Value>,
    expires_at: Instant,
}

pub struct ExactCache {
    entries: RwLock<HashMap<String, CacheEntry>>,
    default_ttl: Duration,
}

impl ExactCache {
    pub fn new(default_ttl: Duration) -> Self {
        Self {
            entries: RwLock::new(HashMap::new()),
            default_ttl,
        }
    }

    /// Generates a unique key based on serialized message history and validation schema
    fn make_key(&self, messages_json: &str, schema_json: &str) -> String {
        format!("{}_||_{}", messages_json, schema_json)
    }

    /// Queries the cache for a hit. Returns `None` if missed or expired.
    pub fn get(&self, messages_json: &str, schema_json: &str) -> Option<ExtractResult<Value>> {
        let key = self.make_key(messages_json, schema_json);
        let guard = self.entries.read().unwrap();
        guard.get(&key)
            .filter(|entry| Instant::now() < entry.expires_at)
            .map(|entry| entry.result.clone())
    }

    /// Caches a successful result with an optional time-to-live duration override
    pub fn set(&self, messages_json: &str, schema_json: &str, result: ExtractResult<Value>, ttl: Option<Duration>) {
        let key = self.make_key(messages_json, schema_json);
        let expires_at = Instant::now() + ttl.unwrap_or(self.default_ttl);
        let mut guard = self.entries.write().unwrap();
        guard.insert(key, CacheEntry { result, expires_at });
    }

    /// Prunes expired entries to free memory
    pub fn prune(&self) {
        let mut guard = self.entries.write().unwrap();
        let now = Instant::now();
        guard.retain(|_, entry| now < entry.expires_at);
    }
}
