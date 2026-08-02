import { Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard';
import NewJob from './pages/NewJob';
import JobDetail from './pages/JobDetail';
import Chat from './pages/Chat';

export default function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/new" element={<NewJob />} />
        <Route path="/jobs/:id" element={<JobDetail />} />
        <Route path="/jobs/:id/chat" element={<Chat />} />
      </Routes>
    </Layout>
  );
}
