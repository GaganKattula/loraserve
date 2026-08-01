import { Link } from 'react-router-dom';

export default function Layout({ children }) {
  return (
    <>
      <header className="app-header">
        <h1><Link to="/" style={{ color: 'inherit', textDecoration: 'none' }}>LoRA Serve</Link></h1>
        <nav>
          <Link to="/">Dashboard</Link>
          <Link to="/new">New Job</Link>
        </nav>
      </header>
      <main className="page">{children}</main>
    </>
  );
}
