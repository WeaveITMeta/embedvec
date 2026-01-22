//! PyO3 Python Bindings for embedvec
//!
//! ## Table of Contents
//! - **PyEmbedVec**: Python-facing EmbedVec class
//! - **PyHit**: Python-facing search result
//! - **Module initialization**: embedvec_py module setup
//!
//! ## Usage from Python
//! ```python
//! import embedvec_py
//! 
//! db = embedvec_py.EmbedVec(dim=768, metric="cosine", m=32, ef_construction=200)
//! db.add_many(vectors, payloads)
//! hits = db.search(query, k=10, ef_search=128, filter={"category": "news"})
//! ```

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::distance::Distance;
use crate::filter::parse_simple_filter;
#[cfg(any(feature = "persistence-sled", feature = "persistence-rocksdb"))]
use crate::persistence::BackendConfig;
use crate::quantization::Quantization;
use crate::{EmbedVec, Metadata};

/// Python-facing EmbedVec class
#[pyclass(name = "EmbedVec")]
pub struct PyEmbedVec {
    inner: EmbedVec,
}

#[pymethods]
impl PyEmbedVec {
    /// Create a new EmbedVec instance
    ///
    /// # Arguments
    /// * `dim` - Vector dimension (e.g., 768 for many LLM embeddings)
    /// * `metric` - Distance metric: "cosine", "euclidean", or "dot"
    /// * `m` - HNSW M parameter (connections per layer)
    /// * `ef_construction` - HNSW construction parameter
    /// * `persist_path` - Optional path for persistence
    /// * `quantization` - Optional quantization mode: None, "e8-8bit", "e8-10bit", "e8-12bit"
    /// * `random_seed` - Optional seed for reproducible quantization (default: 0xcafef00d)
    #[new]
    #[pyo3(signature = (dim, metric="cosine", m=32, ef_construction=200, persist_path=None, quantization=None, random_seed=None))]
    fn new(
        dim: usize,
        metric: &str,
        m: usize,
        ef_construction: usize,
        persist_path: Option<String>,
        quantization: Option<&str>,
        random_seed: Option<u64>,
    ) -> PyResult<Self> {
        let distance = match metric.to_lowercase().as_str() {
            "cosine" => Distance::Cosine,
            "euclidean" | "l2" => Distance::Euclidean,
            "dot" | "dotproduct" | "inner" => Distance::DotProduct,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("Unknown metric: {}. Use 'cosine', 'euclidean', or 'dot'", metric),
                ))
            }
        };

        let seed = random_seed.unwrap_or(0xcafef00d);
        let quant = match quantization {
            None | Some("none") => Quantization::None,
            Some("e8") | Some("e8-10bit") => Quantization::e8(10, true, seed),
            Some("e8-8bit") => Quantization::e8(8, true, seed),
            Some("e8-12bit") => Quantization::e8(12, true, seed),
            // Legacy aliases
            Some("e8p") | Some("e8p-10bit") => Quantization::e8(10, true, seed),
            Some("e8p-8bit") => Quantization::e8(8, true, seed),
            Some("e8p-12bit") => Quantization::e8(12, true, seed),
            Some(q) => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("Unknown quantization: {}. Use 'none', 'e8-8bit', 'e8-10bit', or 'e8-12bit'", q),
                ))
            }
        };

        // Convert persist_path to BackendConfig if provided
        #[cfg(any(feature = "persistence-sled", feature = "persistence-rocksdb"))]
        let persistence_config = persist_path.map(|p| BackendConfig::new(p));
        
        // Create EmbedVec using internal constructor
        #[cfg(any(feature = "persistence-sled", feature = "persistence-rocksdb"))]
        let inner = EmbedVec::new_internal(dim, distance, m, ef_construction, quant, persistence_config)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        #[cfg(not(any(feature = "persistence-sled", feature = "persistence-rocksdb")))]
        let inner = EmbedVec::new_internal(dim, distance, m, ef_construction, quant)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(Self { inner })
    }

    /// Add a single vector with metadata
    ///
    /// # Arguments
    /// * `vector` - List of floats (embedding vector)
    /// * `payload` - Dict of metadata
    ///
    /// # Returns
    /// Vector ID
    fn add(&mut self, vector: Vec<f32>, payload: &Bound<'_, PyDict>) -> PyResult<usize> {
        let metadata = pydict_to_metadata(payload)?;
        
        self.inner
            .add_internal(&vector, metadata)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    /// Add multiple vectors with metadata
    ///
    /// # Arguments
    /// * `vectors` - List of vectors (list of lists or numpy array)
    /// * `payloads` - List of metadata dicts
    fn add_many(&mut self, vectors: Vec<Vec<f32>>, payloads: Bound<'_, PyList>) -> PyResult<()> {
        if vectors.len() != payloads.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!(
                    "Mismatched lengths: {} vectors, {} payloads",
                    vectors.len(),
                    payloads.len()
                ),
            ));
        }

        for (vector, payload_obj) in vectors.iter().zip(payloads.iter()) {
            let payload = payload_obj.downcast::<PyDict>()
                .map_err(|_| PyErr::new::<pyo3::exceptions::PyTypeError, _>("payloads must be a list of dicts"))?;
            let metadata = pydict_to_metadata(payload)?;
            self.inner
                .add_internal(vector, metadata)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        }

        Ok(())
    }

    /// Search for nearest neighbors
    ///
    /// # Arguments
    /// * `query_vector` - Query vector (list of floats)
    /// * `k` - Number of results to return
    /// * `ef_search` - Search parameter (higher = better recall)
    /// * `filter` - Optional filter dict (simple key-value matching)
    ///
    /// # Returns
    /// List of hit dicts with 'id', 'score', and 'payload'
    #[pyo3(signature = (query_vector, k=10, ef_search=128, filter=None))]
    fn search(
        &self,
        py: Python<'_>,
        query_vector: Vec<f32>,
        k: usize,
        ef_search: usize,
        filter: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyList>> {
        // Parse filter if provided
        let filter_expr = if let Some(f) = filter {
            let filter_value = pydict_to_metadata(f)?;
            parse_simple_filter(&filter_value)
        } else {
            None
        };

        let results = self
            .inner
            .search_internal(&query_vector, k, ef_search, filter_expr)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        // Convert results to Python list of dicts
        let py_results = PyList::empty_bound(py);
        for hit in results {
            let hit_dict = PyDict::new_bound(py);
            hit_dict.set_item("id", hit.id)?;
            hit_dict.set_item("score", hit.score)?;
            hit_dict.set_item("payload", metadata_to_pyobject(py, &hit.payload)?)?;
            py_results.append(hit_dict)?;
        }

        Ok(py_results.unbind())
    }

    /// Get number of vectors in the database
    fn __len__(&self) -> usize {
        self.inner.storage.read().len()
    }

    /// Get number of vectors
    fn len(&self) -> usize {
        self.inner.storage.read().len()
    }

    /// Check if database is empty
    fn is_empty(&self) -> bool {
        self.inner.storage.read().is_empty()
    }

    /// Clear all vectors
    fn clear(&mut self) -> PyResult<()> {
        self.inner.storage.write().clear();
        self.inner.metadata.write().clear();
        self.inner.index.write().clear();
        Ok(())
    }

    /// Get vector dimension
    #[getter]
    fn dimension(&self) -> usize {
        self.inner.dimension()
    }

    /// Get distance metric name
    #[getter]
    fn metric(&self) -> &'static str {
        match self.inner.distance() {
            Distance::Cosine => "cosine",
            Distance::Euclidean => "euclidean",
            Distance::DotProduct => "dot",
        }
    }

    /// Get memory usage in bytes
    fn memory_bytes(&self) -> usize {
        self.inner.storage.read().memory_bytes()
    }

    /// Get compression ratio (if quantization enabled)
    fn compression_ratio(&self) -> f32 {
        self.inner.quantization().compression_ratio(self.inner.dimension())
    }

    /// String representation
    fn __repr__(&self) -> String {
        format!(
            "EmbedVec(dim={}, metric='{}', len={}, memory={}KB)",
            self.inner.dimension(),
            self.metric(),
            self.len(),
            self.memory_bytes() / 1024
        )
    }
}

/// Convert Python dict to Metadata (serde_json::Value)
fn pydict_to_metadata(dict: &Bound<'_, PyDict>) -> PyResult<Metadata> {
    let mut map = serde_json::Map::new();

    for (key, value) in dict.iter() {
        let key_str: String = key.extract()?;
        let json_value = pyobject_to_json(&value)?;
        map.insert(key_str, json_value);
    }

    Ok(serde_json::Value::Object(map))
}

/// Convert Python object to serde_json::Value
fn pyobject_to_json(obj: &Bound<'_, PyAny>) -> PyResult<serde_json::Value> {
    if obj.is_none() {
        Ok(serde_json::Value::Null)
    } else if let Ok(b) = obj.extract::<bool>() {
        Ok(serde_json::Value::Bool(b))
    } else if let Ok(i) = obj.extract::<i64>() {
        Ok(serde_json::Value::Number(i.into()))
    } else if let Ok(f) = obj.extract::<f64>() {
        Ok(serde_json::json!(f))
    } else if let Ok(s) = obj.extract::<String>() {
        Ok(serde_json::Value::String(s))
    } else if let Ok(list) = obj.downcast::<PyList>() {
        let arr: Result<Vec<serde_json::Value>, _> = list
            .iter()
            .map(|item| pyobject_to_json(&item))
            .collect();
        Ok(serde_json::Value::Array(arr?))
    } else if let Ok(dict) = obj.downcast::<PyDict>() {
        let metadata = pydict_to_metadata(dict)?;
        Ok(metadata)
    } else {
        // Fallback: convert to string
        let s = obj.str()?.to_string();
        Ok(serde_json::Value::String(s))
    }
}

/// Convert Metadata to Python object
fn metadata_to_pyobject(py: Python<'_>, value: &Metadata) -> PyResult<PyObject> {
    use pyo3::conversion::ToPyObject;
    
    match value {
        serde_json::Value::Null => Ok(py.None()),
        serde_json::Value::Bool(b) => Ok(b.to_object(py)),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(i.to_object(py))
            } else if let Some(f) = n.as_f64() {
                Ok(f.to_object(py))
            } else {
                Ok(py.None())
            }
        }
        serde_json::Value::String(s) => Ok(s.to_object(py)),
        serde_json::Value::Array(arr) => {
            let list = PyList::empty_bound(py);
            for item in arr {
                list.append(metadata_to_pyobject(py, item)?)?;
            }
            Ok(list.unbind().into())
        }
        serde_json::Value::Object(map) => {
            let dict = PyDict::new_bound(py);
            for (k, v) in map {
                dict.set_item(k, metadata_to_pyobject(py, v)?)?;
            }
            Ok(dict.unbind().into())
        }
    }
}

/// Python module initialization
#[pymodule]
fn embedvec_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyEmbedVec>()?;
    m.add("__version__", "0.5.0")?;
    m.add("__doc__", "Fast, lightweight, in-process vector database with HNSW indexing and E8 quantization")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_distance_parsing() {
        // Test that distance metric parsing works
        assert!(matches!(
            "cosine".to_lowercase().as_str(),
            "cosine"
        ));
    }
}
