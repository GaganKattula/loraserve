import { Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard';
import NewJob from './pages/NewJob';
import JobDetail from './pages/JobDetail';
import Inference from './pages/Inference';

export default function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/new" element={<NewJob />} />
        <Route path="/jobs/:id" element={<JobDetail />} />
        <Route path="/jobs/:id/infer" element={<Inference />} />
      </Routes>
    </Layout>
  );
}
