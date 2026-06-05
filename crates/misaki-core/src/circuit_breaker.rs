use std::sync::Mutex;
use std::time::{Duration, Instant};

#[derive(Debug)]
pub struct CircuitBreaker {
    failures: Mutex<usize>,
    last_failure: Mutex<Option<Instant>>,
    threshold: usize,
    cooldown: Duration,
}

impl CircuitBreaker {
    pub fn new(threshold: usize, cooldown: Duration) -> Self {
        Self {
            failures: Mutex::new(0),
            last_failure: Mutex::new(None),
            threshold,
            cooldown,
        }
    }

    /// Returns true if the circuit breaker is currently open (tripped)
    pub fn is_open(&self) -> bool {
        let failures = self.failures.lock().unwrap();
        if *failures >= self.threshold {
            let last_fail = self.last_failure.lock().unwrap();
            if last_fail.is_some_and(|instant| instant.elapsed() < self.cooldown) {
                return true; // Cooldown period has not expired yet
            }
        }
        false
    }

    /// Resets the failure count on a successful call
    pub fn record_success(&self) {
        let mut failures = self.failures.lock().unwrap();
        *failures = 0;
        let mut last_fail = self.last_failure.lock().unwrap();
        *last_fail = None;
    }

    /// Increments the failure count and records the failure timestamp
    pub fn record_failure(&self) {
        let mut failures = self.failures.lock().unwrap();
        *failures += 1;
        let mut last_fail = self.last_failure.lock().unwrap();
        *last_fail = Some(Instant::now());
    }
}
