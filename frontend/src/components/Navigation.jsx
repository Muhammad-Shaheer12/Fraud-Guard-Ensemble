import { Link, useLocation } from 'react-router-dom';

export default function Navigation() {
  const location = useLocation();

  return (
    <nav className="navbar">
      <div className="nav-brand">
        <h2>FraudGuard <span>AI</span></h2>
      </div>
      <div className="nav-links">
        <Link to="/" className={location.pathname === '/' ? 'active' : ''}>
          Single Model Demo
        </Link>
        <Link to="/ensemble" className={location.pathname === '/ensemble' ? 'active' : ''}>
          Ensemble Analysis
        </Link>
        <Link to="/analytics" className={location.pathname === '/analytics' ? 'active' : ''}>
          Training Analytics
        </Link>
      </div>
    </nav>
  );
}
