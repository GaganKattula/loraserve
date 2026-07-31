import { useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { createJob } from '../api';

export default function NewJob() {
  const [taskDesc, setTaskDesc] = useState('');
  const [examples, setExamples] = useState(null);
  const [fileName, setFileName] = useState('');
  const [parseError, setParseError] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState('');
  const fileRef = useRef(null);
  const navigate = useNavigate();

  const handleFile = (file) => {
    if (!file) return;
    setFileName(file.name);
    setParseError('');
    setExamples(null);

    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const lines = e.target.result.trim().split('\n');
        const parsed = lines.map((line, i) => {
          const obj = JSON.parse(line);
          if (!obj.text || !obj.label) {
            throw new Error(`Line ${i + 1}: missing "text" or "label" field`);
          }
          return { text: obj.text, label: obj.label };
        });
        if (parsed.length < 20) {
          setParseError(`Only ${parsed.length} examples found. Need at least 20.`);
          return;
        }
        setExamples(parsed);
      } catch (err) {
        setParseError(err.message);
      }
    };
    reader.readAsText(file);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  };

  const handleSubmit = async () => {
    if (!taskDesc.trim() || !examples) return;
    setSubmitting(true);
    setSubmitError('');
    try {
      const data = await createJob(taskDesc.trim(), examples);
      navigate(`/jobs/${data.job_id}`);
    } catch (err) {
      setSubmitError(err.message);
      setSubmitting(false);
    }
  };

  return (
    <>
      <div className="page-title">New Training Job</div>

      <div className="card">
        <div className="form-group">
          <label>Task Description</label>
          <input
            type="text"
            placeholder="e.g. Classify customer emails as spam or not"
            value={taskDesc}
            onChange={(e) => setTaskDesc(e.target.value)}
          />
        </div>

        <div className="form-group">
          <label>Dataset (.jsonl)</label>
          <div
            className={`file-drop ${examples ? 'has-file' : ''}`}
            onClick={() => fileRef.current?.click()}
            onDrop={handleDrop}
            onDragOver={(e) => e.preventDefault()}
          >
            {examples
              ? `${fileName} - ${examples.length} examples ready`
              : 'Click or drop a .jsonl file here'}
            <br />
            <span style={{ fontSize: '12px' }}>
              Each line: {`{"text": "...", "label": "..."}`}
            </span>
          </div>
          <input
            ref={fileRef}
            type="file"
            accept=".jsonl,.json"
            style={{ display: 'none' }}
            onChange={(e) => handleFile(e.target.files[0])}
          />
          {parseError && <div className="error-text">{parseError}</div>}
        </div>

        {submitError && <div className="error-text" style={{ marginBottom: '12px' }}>{submitError}</div>}

        <button
          className="btn btn-primary"
          disabled={!taskDesc.trim() || !examples || submitting}
          onClick={handleSubmit}
        >
          {submitting ? 'Submitting...' : 'Start Training'}
        </button>
      </div>

      {examples && (
        <div className="card">
          <div style={{ fontSize: '13px', fontWeight: 600, marginBottom: '8px' }}>Preview (first 5 examples)</div>
          <table>
            <thead>
              <tr><th>Text</th><th>Label</th></tr>
            </thead>
            <tbody>
              {examples.slice(0, 5).map((ex, i) => (
                <tr key={i} style={{ cursor: 'default' }}>
                  <td style={{ maxWidth: '500px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{ex.text}</td>
                  <td className="mono">{ex.label}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </>
  );
}
