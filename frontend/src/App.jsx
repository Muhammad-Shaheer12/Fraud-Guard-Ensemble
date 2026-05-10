import { useState } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navigation from './components/Navigation';
import VerificationPage from './pages/VerificationPage';
import EnsemblePage from './pages/EnsemblePage';
import AnalyticsPage from './pages/AnalyticsPage';
import './App.css';

const initialForm = {
  transactionId: '',
  userId: '',
  merchantId: '',
  amount: '',
  hour: '',
};

export default function App() {
  const [form, setForm] = useState(initialForm);

  const onField = (event) => {
    const { name, value } = event.target;
    setForm((prev) => ({ ...prev, [name]: value }));
  };

  const resetForm = () => {
    setForm(initialForm);
  };

  return (
    <Router>
      <div className="app-container">
        <Navigation />
        <Routes>
          <Route path="/" element={<VerificationPage form={form} onField={onField} resetFormApp={resetForm} />} />
          <Route path="/ensemble" element={<EnsemblePage form={form} onField={onField} resetFormApp={resetForm} />} />
          <Route path="/analytics" element={<AnalyticsPage />} />
        </Routes>
      </div>
    </Router>
  );
}
