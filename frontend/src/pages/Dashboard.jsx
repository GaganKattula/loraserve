import { useState, useEffect } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { listJobs } from '../api';
import StatusBadge from '../components/StatusBadge';

export default function Dashboard() {
  const [jobs, setJobs] = useState([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();

  const fetchJobs = async () => {
    try {
      const data = await listJobs(20, 0);
      setJobs(data.jobs);
      setTotal(data.total);
    } catch {
      // API may not be running
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchJobs();
    const interval = setInterval(fetchJobs, 5000);
    return () => clearInterval(interval);
  }, []);

  const fmtDate = (iso) => {
    if (!iso) return '-';
    return new Date(iso).toLocaleString(undefined, {
      month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
    });
  };

  return (
    <>
      <div className="page-title">
        <span>Jobs</span>
        <Link to="/new" className="btn btn-primary" style={{ marginLeft: 'auto', fontSize: '13px' }}>
          + New Job
        </Link>
      </div>

      {loading ? (
        <div className="empty-state">Loading...</div>
      ) : jobs.length === 0 ? (
        <div className="empty-state">
          No jobs yet. <Link to="/new">Submit your first training job.</Link>
        </div>
      ) : (
        <>
          <table>
            <thead>
              <tr>
                <th>Status</th>
                <th>Task</th>
                <th>Examples</th>
                <th>Rank</th>
                <th>Eval Loss</th>
                <th>Created</th>
              </tr>
            </thead>
            <tbody>
              {jobs.map((job) => (
                <tr key={job.job_id} onClick={() => navigate(`/jobs/${job.job_id}`)}>
                  <td><StatusBadge status={job.status} /></td>
                  <td>{job.task_description || '-'}</td>
                  <td className="mono">{job.num_examples ?? '-'}</td>
                  <td className="mono">{job.lora_r ? `r${job.lora_r}` : '-'}</td>
                  <td className="mono">{job.eval_loss != null ? job.eval_loss.toFixed(4) : '-'}</td>
                  <td>{fmtDate(job.created_at)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div style={{ fontSize: '12px', color: '#888', marginTop: '8px' }}>
            {total} job{total !== 1 ? 's' : ''} total
          </div>
        </>
      )}
    </>
  );
}
