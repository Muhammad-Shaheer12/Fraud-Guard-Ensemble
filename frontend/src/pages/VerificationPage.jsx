import { useMemo, useState } from 'react';
import axios from 'axios';

const API_BASE = 'http://127.0.0.1:8000';

// initialForm is now in App.jsx

export default function VerificationPage({ form, onField, resetFormApp }) {
  const [modelName, setModelName] = useState('cnn');
  // Form state is now managed by App.jsx
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');

  const healthMetrics = useMemo(() => {
    return [
      { label: 'Engine', value: modelName.toUpperCase() },
      { label: 'API', value: 'Online Ready' },
      { label: 'Queue', value: loading ? 'Processing' : 'Idle' },
    ];
  }, [modelName, loading]);

  // onField is now handled by App.jsx

  const resetForm = () => {
    resetFormApp();
    setResult(null);
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

  const verifyTransaction = async () => {
    setError('');
    setResult(null);

    if (!form.transactionId.trim()) {
      setError('Transaction ID is required.');
      return;
    }

    setLoading(true);
    try {
      const response = await axios.post(`${API_BASE}/verify-transaction`, {
        model_name: modelName,
        transaction_id: form.transactionId,
        user_identifier: form.userId || null,
        merchant_identifier: form.merchantId || null,
        feature_vector: buildFeatureVector(),
      });
      setResult(response.data);
    } catch (err) {
      setError(err?.response?.data?.detail || 'Prediction request failed.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page-content">
      <header className="page-header">
        <h1>Single Model Verification</h1>
        <p>Test individual architectures and view specific feature attributions.</p>
      </header>

      <main className="layout">
        <section className="panel dashboard">
          <h2>Dashboard</h2>
          <div className="metric-grid">
            {healthMetrics.map((metric) => (
              <article className="metric" key={metric.label}>
                <span>{metric.label}</span>
                <strong>{metric.value}</strong>
              </article>
            ))}
          </div>
        </section>

        <section className="panel verification">
          <h2>Transaction Verification</h2>
          <label>
            Model Selection
            <select value={modelName} onChange={(e) => setModelName(e.target.value)}>
              <option value="cnn">CNN</option>
              <option value="lstm">LSTM</option>
              <option value="transformer">Transformer</option>
              <option value="hybrid">Hybrid</option>
            </select>
          </label>

          <div className="field-grid">
            <label>
              Transaction ID
              <input name="transactionId" value={form.transactionId} onChange={onField} />
            </label>
            <label>
              User ID
              <input name="userId" value={form.userId} onChange={onField} />
            </label>
            <label>
              Merchant ID
              <input name="merchantId" value={form.merchantId} onChange={onField} />
            </label>
            <label>
              Amount
              <input name="amount" type="number" value={form.amount} onChange={onField} />
            </label>
            <label>
              Transaction Hour
              <input name="hour" type="number" value={form.hour} onChange={onField} />
            </label>
          </div>

          <div className="actions">
            <button className="cta" disabled={loading} onClick={verifyTransaction}>
              {loading ? 'Verifying...' : 'Verify Transaction'}
            </button>
            <button className="ghost" onClick={resetForm}>Reset Form</button>
          </div>

          {error && <p className="error">{error}</p>}
        </section>

        <section className="panel analytics">
          <h2>Analytics & Interpretability</h2>
          {!result && <p className="subtle">Run a verification to view confidence and interpretability summary.</p>}

          {result && (
            <div className="result-card">
              <p className={result.prediction === 'Fraud' ? 'fraud' : 'safe'}>
                {result.prediction}
              </p>
              <p>Confidence: {(result.confidence * 100).toFixed(2)}%</p>
              <p>Fraud Probability: {(result.fraud_probability * 100).toFixed(2)}%</p>
              <p>Method: {result.interpretability?.method}</p>
              <p>Top Feature Indices: {(result.interpretability?.top_feature_indices || []).join(', ')}</p>
            </div>
          )}
        </section>
      </main>
    </div>
  );
}
