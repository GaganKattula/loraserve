import { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import { getJob } from '../api';
import StatusBadge from '../components/StatusBadge';
import LossChart from '../components/LossChart';

export default function JobDetail() {
  const { id } = useParams();
  const [job, setJob] = useState(null);
  const [error, setError] = useState('');
  const [chartDone, setChartDone] = useState(false);

  useEffect(() => {
    getJob(id).then(setJob).catch((e) => setError(e.message));
  }, [id]);

  // Re-fetch job when chart signals done to get final metadata
  const handleStatusChange = (status) => {
    if (status === 'done') {
      setChartDone(true);
      getJob(id).then(setJob).catch(() => {});
    }
  };

  if (error) return <div className="empty-state">Error: {error}</div>;
  if (!job) return <div className="empty-state">Loading...</div>;

  const showChart = job.status === 'running' || job.status === 'queued' || job.status === 'complete';

  return (
    <>
      <div className="page-title">
        <StatusBadge status={job.status} />
        <span style={{ fontSize: '14px', color: '#888' }} className="mono">{id}</span>
      </div>

      {/* Metadata panel */}
      <div className="card">
        <div className="meta-grid">
          <div className="meta-item">
            <div className="label">Task</div>
            <div className="value">{job.task_description || '-'}</div>
          </div>
          <div className="meta-item">
            <div className="label">Examples</div>
            <div className="value">{job.num_examples ?? '-'}</div>
          </div>
          <div className="meta-item">
            <div className="label">LoRA Rank</div>
            <div className="value">{job.lora_r ? `r=${job.lora_r}, alpha=${job.lora_alpha}` : 'pending'}</div>
          </div>
          <div className="meta-item">
            <div className="label">Target Modules</div>
            <div className="value">{job.target_modules || 'pending'}</div>
          </div>
          <div className="meta-item">
            <div className="label">Adapter Size</div>
            <div className="value">{job.adapter_size_mb ? `${job.adapter_size_mb.toFixed(1)} MB` : 'pending'}</div>
          </div>
          <div className="meta-item">
            <div className="label">Eval Loss</div>
            <div className="value">{job.eval_loss != null ? job.eval_loss.toFixed(4) : 'pending'}</div>
          </div>
          <div className="meta-item">
            <div className="label">Created</div>
            <div className="value">{job.created_at ? new Date(job.created_at).toLocaleString() : '-'}</div>
          </div>
        </div>
      </div>

      {/* Error message for failed jobs */}
      {job.status === 'failed' && job.error_message && (
        <div className="card" style={{ borderLeft: '4px solid #dc2626' }}>
          <div style={{ fontSize: '13px', fontWeight: 600, color: '#991b1b', marginBottom: '4px' }}>Error</div>
          <div className="mono" style={{ fontSize: '13px' }}>{job.error_message}</div>
        </div>
      )}

      {/* Training chart */}
      {showChart && (
        <div className="card">
          <div style={{ fontSize: '13px', fontWeight: 600, marginBottom: '8px' }}>Training Progress</div>
          <LossChart jobId={id} onStatusChange={handleStatusChange} />
        </div>
      )}

      {/* Inference link for completed jobs */}
      {job.status === 'complete' && (
        <Link to={`/jobs/${id}/chat`} className="btn btn-primary" style={{ display: 'inline-block', marginTop: '8px' }}>
          Chat with Model
        </Link>
      )}
    </>
  );
}
