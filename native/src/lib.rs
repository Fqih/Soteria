//! Native speedups for the Avo agent runtime.
//!
//! The single export today is ``cache_key_hash`` — the SHA-256-based
//! digest used by ``avo.providers.prompt_cache.cache_key_for_request``
//! to derive an OpenAI-style ``prompt_cache_key`` per turn.
//!
//! The Rust side collapses the per-message Python serialisation into
//! one streaming pass with no intermediate allocations, which
//! dominates cache-key cost on large conversations. The pure-Python
//! fallback in ``avo.providers.prompt_cache`` stays authoritative so
//! behaviour never depends on the native module being importable.

use pyo3::prelude::*;
use sha2::{Digest, Sha256};

/// Compute the ``avo:<run_id>:<step>:<hex16>`` cache key for a list of
/// already-serialised message fragments.
///
/// Parameters mirror :func:`avo.providers.prompt_cache.cache_key_for_request`:
/// * ``run_id`` — opaque identifier for the run.
/// * ``step`` — monotonic turn counter.
/// * ``messages_json`` — ``list[str]`` where each entry is a canonical
///   representation of one message (``repr(sorted_dict)`` is fine).
#[pyfunction]
fn cache_key_hash(run_id: &str, step: u64, messages_json: Vec<String>) -> String {
    let mut hasher = Sha256::new();
    // Domain-separate the run_id + step from the message body so two
    // runs with identical messages still produce different keys.
    hasher.update(run_id.as_bytes());
    hasher.update(b"\0");
    hasher.update(step.to_le_bytes());
    hasher.update(b"\0");
    for fragment in messages_json {
        hasher.update(fragment.as_bytes());
        hasher.update(b"\x1f"); // ASCII unit separator — collision-free
    }
    let digest = hasher.finalize();
    let mut out = String::with_capacity(4 + run_id.len() + 20 + 16);
    out.push_str("avo:");
    out.push_str(run_id);
    out.push(':');
    let step_str = step.to_string();
    out.push_str(&step_str);
    out.push(':');
    out.push_str(&hex::encode(&digest[..8]));
    out
}

/// Return the package version baked into the native module.
#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// PyO3 module declaration.
#[pymodule]
fn avo_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(cache_key_hash, m)?)?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    Ok(())
}
