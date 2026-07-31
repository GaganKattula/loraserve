import { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import { getJob, runInference } from '../api';
import StatusBadge from '../components/StatusBadge';

export default function Inference() {
  const { id } = useParams();
  const [job, setJob] = useState(null);
  const [prompt, setPrompt] = useState('');
  const [maxTokens, setMaxTokens] = useState(20);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    getJob(id).then(setJob).catch((e) => setError(e.message));
  }, [id]);

  const handleRun = async () => {
    if (!prompt.trim()) return;
    setLoading(true);
    setError('');
    setResult(null);
    try {
      const data = await runInference(id, prompt.trim(), maxTokens);
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  if (!job && !error) return <div className="empty-state">Loading...</div>;

  return (
    <>
      <div className="page-title">
        <span>Inference</span>
        <Link to={`/jobs/${id}`} style={{ fontSize: '13px', color: '#888', marginLeft: '8px' }}>
          back to job
        </Link>
      </div>

      {/* Job summary */}
      {job && (
        <div className="card">
          <div className="meta-grid">
            <div className="meta-item">
              <div className="label">Status</div>
              <StatusBadge status={job.status} />
            </div>
            <div className="meta-item">
              <div className="label">LoRA</div>
              <div className="value">{job.lora_r ? `r=${job.lora_r}` : '-'}</div>
            </div>
            <div className="meta-item">
              <div className="label">Eval Loss</div>
              <div className="value">{job.eval_loss != null ? job.eval_loss.toFixed(4) : '-'}</div>
            </div>
            <div className="meta-item">
              <div className="label">Adapter</div>
              <div className="value">{job.adapter_size_mb ? `${job.adapter_size_mb.toFixed(1)} MB` : '-'}</div>
            </div>
          </div>
        </div>
      )}

      {/* Prompt input */}
      <div className="card">
        <div className="form-group">
          <label>Prompt</label>
          <textarea
            placeholder="Enter text for the model to process..."
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Enter' && e.metaKey) handleRun(); }}
          />
        </div>

        <div className="form-group" style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <label style={{ margin: 0, whiteSpace: 'nowrap' }}>Max tokens: {maxTokens}</label>
          <input
            type="range"
            min="5"
            max="200"
            value={maxTokens}
            onChange={(e) => setMaxTokens(Number(e.target.value))}
            style={{ flex: 1 }}
          />
        </div>

        <button
          className="btn btn-primary"
          disabled={!prompt.trim() || loading || job?.status !== 'complete'}
          onClick={handleRun}
        >
          {loading ? 'Running...' : 'Run Inference'}
        </button>

        {job?.status !== 'complete' && (
          <span style={{ fontSize: '12px', color: '#888', marginLeft: '12px' }}>
            Job must be complete to run inference
          </span>
        )}
      </div>

      {/* Error */}
      {error && (
        <div className="card" style={{ borderLeft: '4px solid #dc2626' }}>
          <div className="error-text">{error}</div>
        </div>
      )}

      {/* Result */}
      {result && (
        <div className="card">
          <div style={{ fontSize: '13px', fontWeight: 600, marginBottom: '8px' }}>Output</div>
          <div className="infer-output">{result.output}</div>
          <div className="infer-meta">
            <span>Latency: {result.latency_ms}ms</span>
            <span>Cache: {result.cache_hit ? 'HIT' : 'MISS'}</span>
          </div>
        </div>
      )}
    </>
  );
}
