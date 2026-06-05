use std::collections::HashMap;
use serde_json::Value;
use crate::error::{MisakiError, Result};

/// Cleans raw text from LLM which might be wrapped in ```json ... ``` markdown blocks.
pub fn clean_json_string(raw: &str) -> String {
    let trimmed = raw.trim();
    if trimmed.starts_with("```") {
        // Find the start of the actual content
        let mut lines = trimmed.lines();
        let first_line = lines.next().unwrap_or("");
        let content_lines: Vec<&str> = if first_line.starts_with("```json") || first_line.starts_with("```") {
            lines.collect()
        } else {
            trimmed.lines().collect()
        };
        
        let mut cleaned = content_lines.join("\n");
        if cleaned.ends_with("```") {
            cleaned.truncate(cleaned.len() - 3);
        }
        cleaned.trim().to_string()
    } else {
        trimmed.to_string()
    }
}

/// Helper to parse raw content as JSON, cleaning markdown blocks if necessary.
pub fn parse_json(raw: &str) -> Result<Value> {
    let cleaned = clean_json_string(raw);
    serde_json::from_str(&cleaned).map_err(|e| {
        MisakiError::Validation(format!("Failed to parse JSON: {} (raw output: {})", e, raw))
    })
}

/// Scores field-level confidence.
/// Returns a map of field name -> confidence score (0.0 to 1.0).
pub fn score_confidence(
    value: &Value,
    attempts_needed: usize,
    did_repair: bool,
    schema_valid: bool,
    logprobs: Option<&[crate::providers::LogprobInfo]>,
) -> HashMap<String, f32> {
    let mut scores = HashMap::new();
    
    // Base confidence based on retries and repairs
    let mut base_conf = if !schema_valid {
        0.0
    } else if attempts_needed == 1 {
        0.98
    } else if did_repair {
        0.85 - (attempts_needed as f32 * 0.05)
    } else {
        0.90 - (attempts_needed as f32 * 0.04)
    };
    
    // Blend with actual logprob certainty if available
    if schema_valid {
        let avg_prob = logprobs
            .filter(|l| !l.is_empty())
            .map(|lps| {
                let sum_prob: f32 = lps.iter().map(|lp| lp.logprob.exp()).sum();
                sum_prob / lps.len() as f32
            });
            
        if let Some(avg) = avg_prob {
            base_conf = (base_conf + avg) / 2.0;
        }
    }
    
    let base_conf = base_conf.clamp(0.01, 1.0);

    if let Value::Object(map) = value {
        for key in map.keys() {
            scores.insert(key.clone(), base_conf);
        }
    } else {
        scores.insert("root".to_string(), base_conf);
    }

    scores
}

/// Searches the source text for evidence of the extracted field values.
/// This matches Section 10 (Evidence-Based Extraction).
pub fn extract_evidence(value: &Value, source_text: &str) -> HashMap<String, String> {
    let mut evidence_map = HashMap::new();
    if let Value::Object(map) = value {
        for (key, val) in map {
            let val_str = match val {
                Value::String(s) => s.clone(),
                Value::Number(n) => n.to_string(),
                Value::Bool(b) => b.to_string(),
                _ => continue, // Skip complex nested fields for simple evidence string matching
            };

            // Simple substring matching for evidence
            if !val_str.is_empty() && source_text.contains(&val_str) {
                // Find context lines
                if let Some(idx) = source_text.find(&val_str) {
                    let start = idx.saturating_sub(20);
                    let end = std::cmp::min(source_text.len(), idx + val_str.len() + 20);
                    let context = &source_text[start..end];
                    evidence_map.insert(key.clone(), format!("...{}...", context.replace('\n', " ").trim()));
                }
            } else {
                evidence_map.insert(key.clone(), "Not explicitly found in source text".to_string());
            }
        }
    }
    evidence_map
}

/// Helper to parse partial JSON during streaming by auto-closing brackets, braces, and strings.
pub fn parse_partial_json(raw: &str) -> Option<Value> {
    let cleaned = clean_json_string(raw);
    if cleaned.is_empty() {
        return None;
    }
    
    // Try parsing as-is
    if let Ok(val) = serde_json::from_str(&cleaned) {
        return Some(val);
    }
    
    // Helper to attempt to close open structures
    let close_json = |text: &str| -> Option<Value> {
        let mut stack = Vec::new();
        let mut in_string = false;
        let mut escaped = false;
        
        for c in text.chars() {
            if in_string {
                if escaped {
                    escaped = false;
                } else if c == '\\' {
                    escaped = true;
                } else if c == '"' {
                    in_string = false;
                }
            } else {
                match c {
                    '"' => in_string = true,
                    '{' => stack.push('}'),
                    '[' => stack.push(']'),
                    '}' if stack.last() == Some(&'}') => {
                        stack.pop();
                    }
                    ']' if stack.last() == Some(&']') => {
                        stack.pop();
                    }
                    _ => {}
                }
            }
        }
        
        let mut closed = text.to_string();
        if in_string {
            closed.push('"');
        }
        for &c in stack.iter().rev() {
            closed.push(c);
        }
        serde_json::from_str(&closed).ok()
    };
    
    // Try directly closing the text
    if let Some(val) = close_json(&cleaned) {
        return Some(val);
    }
    
    // If that fails, try truncating the text at commas, colons, or boundaries
    // from the right to strip incomplete key-value pairs.
    let mut current = cleaned;
    while !current.is_empty() {
        if let Some(last_idx) = current.rfind([',', '{', '[', '}']) {
            current.truncate(last_idx);
            if let Some(val) = close_json(&current) {
                return Some(val);
            }
        } else {
            break;
        }
    }
    
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_parse_partial_json() {
        // Complete JSON
        assert_eq!(
            parse_partial_json(r#"{"invoice_number": "INV-123", "amount": 100.0}"#),
            Some(json!({"invoice_number": "INV-123", "amount": 100.0}))
        );

        // Incomplete key/value
        assert_eq!(
            parse_partial_json(r#"{"invoice_number": "INV-123", "amount":"#),
            Some(json!({"invoice_number": "INV-123"}))
        );

        // Trailing comma
        assert_eq!(
            parse_partial_json(r#"{"invoice_number": "INV-123","#),
            Some(json!({"invoice_number": "INV-123"}))
        );

        // Partial number value
        assert_eq!(
            parse_partial_json(r#"{"invoice_number": "INV-123", "amount": 12"#),
            Some(json!({"invoice_number": "INV-123", "amount": 12}))
        );

        // Partial string value
        assert_eq!(
            parse_partial_json(r#"{"invoice_number": "INV-12"#),
            Some(json!({"invoice_number": "INV-12"}))
        );

        // Empty input
        assert_eq!(parse_partial_json(""), None);
        assert_eq!(parse_partial_json("{"), Some(json!({})));
    }
}
