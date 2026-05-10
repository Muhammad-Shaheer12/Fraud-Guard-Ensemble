import { useState } from 'react';
import axios from 'axios';

const API_BASE = 'http://127.0.0.1:8000';

// initialForm is now in App.jsx

export default function EnsemblePage({ form, onField, resetFormApp }) {
  // Form state is now managed by App.jsx
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState('');

  // onField is now handled by App.jsx

  const resetForm = () => {
    resetFormApp();
    setResults(null);
    setError('');
  };

  const buildFeatureVector = () => {
    const amount = Number(form.amount || 0);
    const hour = Number(form.hour || 0);
    const accountSignal = form.userId.length % 11;
    const merchantSignal = form.merchantId.length % 13;
    const interaction = amount * 0.01 + hour * 0.1;
    return [amount, hour, accountSignal, merchantSignal, interaction];
  };

  const verifyEnsemble = async () => {
    setError('');
    setResults(null);

    if (!form.transactionId.trim()) {
      setError('Transaction ID is required.');
      return;
    }

    setLoading(true);
    try {
      const response = await axios.post(`${API_BASE}/verify-ensemble`, {
        transaction_id: form.transactionId,
        user_identifier: form.userId || null,
        merchant_identifier: form.merchantId || null,
        feature_vector: buildFeatureVector(),
        model_name: "dummy" // not used in ensemble but required by schema
      });
      setResults(response.data);
    } catch (err) {
      setError(err?.response?.data?.detail || 'Ensemble prediction request failed.');
    } finally {
      setLoading(false);
    }
  };

  const getModelRationale = (modelName, prediction) => {
    switch (modelName) {
      case 'cnn':
        return "CNN detects localized spatial correlations within the feature vector. A strong signal indicates clustered anomalies in transaction metadata.";
      case 'lstm':
        return "LSTM treats the 2,803 features as a sequential manifold. It looks for recursive dependencies and irregular sequencing in the transaction profile.";
      case 'transformer':
        return "Transformer applies global self-attention across all features simultaneously. It is highly sensitive to non-linear, distant feature interactions.";
      case 'hybrid':
        return "Hybrid aggregates CNN spatial features with Transformer attention mechanisms, serving as the ultimate ensemble tie-breaker to mitigate individual blind spots.";
      default:
        return "Evaluated using standard neural activations.";
    }
  };

  return (
    <div className="page-content">
      <header className="page-header">
        <h1>Model Ensemble Analysis</h1>
        <p>Run a transaction across all architectures simultaneously to visualize model disagreement.</p>
      </header>

      <main className="layout ensemble-layout">
        <section className="panel verification">
          <h2>Transaction Input</h2>
          <div className="field-grid">
            <label>Transaction ID<input name="transactionId" value={form.transactionId} onChange={onField} /></label>
            <label>User ID<input name="userId" value={form.userId} onChange={onField} /></label>
            <label>Merchant ID<input name="merchantId" value={form.merchantId} onChange={onField} /></label>
            <label>Amount<input name="amount" type="number" value={form.amount} onChange={onField} /></label>
            <label>Transaction Hour<input name="hour" type="number" value={form.hour} onChange={onField} /></label>
          </div>
          <div className="actions">
            <button className="cta" disabled={loading} onClick={verifyEnsemble}>
              {loading ? 'Evaluating...' : 'Run Full Ensemble Evaluation'}
            </button>
            <button className="ghost" onClick={resetForm}>Reset</button>
          </div>
          {error && <p className="error">{error}</p>}
        </section>

        <section className="panel ensemble-results">
          <h2>Ensemble Output</h2>
          {!results && !loading && <p className="subtle">Awaiting transaction evaluation...</p>}
          {loading && <p className="subtle">Broadcasting request to GPU cluster...</p>}
          
          {results && (
            <div className="ensemble-grid">
              {['cnn', 'lstm', 'transformer', 'hybrid'].map((modelName) => {
                const res = results[modelName];
                if (!res) return null;
                
                if (res.error) {
                  return (
                    <div key={modelName} className="result-card error-card">
                      <h3>{modelName.toUpperCase()}</h3>
                      <p>Error: {res.error}</p>
                    </div>
                  );
                }

                return (
                  <div key={modelName} className="result-card">
                    <h3>{modelName.toUpperCase()}</h3>
                    <p className={res.prediction === 'Fraud' ? 'fraud' : 'safe'}>
                      {res.prediction}
                    </p>
                    <p>Confidence: {(res.confidence * 100).toFixed(2)}%</p>
                    <p>Fraud Prob: {(res.fraud_probability * 100).toFixed(2)}%</p>
                    <p className="subtle-small">Top Features: {(res.interpretability?.top_feature_indices || []).join(', ')}</p>
                    <div className="rationale-box" style={{ marginTop: '1rem', padding: '0.75rem', background: 'rgba(0,0,0,0.2)', borderRadius: '6px', fontSize: '0.85rem', color: '#a0a0b0', textAlign: 'left', borderLeft: res.prediction === 'Fraud' ? '3px solid #ff4a4a' : '3px solid #4ade80' }}>
                      <strong>Architectural Rationale:</strong><br/>
                      {getModelRationale(modelName, res.prediction)}
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </section>
      </main>
    </div>
  );
}
