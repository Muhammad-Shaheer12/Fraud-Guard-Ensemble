const API_BASE = 'http://127.0.0.1:8000';

export default function AnalyticsPage() {
  const models = ['cnn', 'lstm', 'transformer', 'hybrid'];

  return (
    <div className="page-content">
      <header className="page-header">
        <h1>Training Analytics & Performance</h1>
        <p>Review the comprehensive evaluation metrics and visualizations from the training pipeline.</p>
      </header>

      <main className="layout analytics-layout">
        <section className="panel full-width">
          <h2>Global Model Comparison</h2>
          <div className="gallery-grid">
            <div className="gallery-item large">
              <h3>ROC Overlay</h3>
              <p className="subtle-small">Receiver Operating Characteristic across all architectures.</p>
              <img src={`${API_BASE}/plots/roc_overlay.png`} alt="ROC Overlay" />
            </div>
            <div className="gallery-item large">
              <h3>Performance Metrics</h3>
              <p className="subtle-small">F1-Score, AUC, Precision, and Recall comparison.</p>
              <img src={`${API_BASE}/plots/model_comparison.png`} alt="Model Comparison" />
            </div>
          </div>
        </section>

        <section className="panel full-width">
          <h2>Individual Confusion Matrices</h2>
          <div className="gallery-grid">
            {models.map(model => (
              <div key={model} className="gallery-item small">
                <h3>{model.toUpperCase()} Matrix</h3>
                <img src={`${API_BASE}/plots/confusion_matrix_${model}.png`} alt={`${model} Confusion Matrix`} />
              </div>
            ))}
          </div>
        </section>

        <section className="panel full-width">
          <h2>Precision-Recall Curves</h2>
          <div className="gallery-grid">
            {models.map(model => (
              <div key={model} className="gallery-item small">
                <h3>{model.toUpperCase()} PR Curve</h3>
                <img src={`${API_BASE}/plots/precision_recall_${model}.png`} alt={`${model} PR Curve`} />
              </div>
            ))}
          </div>
        </section>
      </main>
    </div>
  );
}
