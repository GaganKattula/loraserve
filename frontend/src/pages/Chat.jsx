import { useState, useEffect, useRef } from 'react';
import { useParams, Link } from 'react-router-dom';
import { getJob, chat } from '../api';
import StatusBadge from '../components/StatusBadge';

export default function Chat() {
  const { id } = useParams();
  const [job, setJob] = useState(null);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [maxTokens, setMaxTokens] = useState(50);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const messagesEndRef = useRef(null);

  useEffect(() => {
    getJob(id).then(setJob).catch((e) => setError(e.message));
  }, [id]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim() || loading) return;

    const userMsg = { role: 'user', content: input.trim() };
    const newMessages = [...messages, userMsg];
    setMessages(newMessages);
    setInput('');
    setLoading(true);
    setError('');

    try {
      const data = await chat(id, newMessages, maxTokens);
      const assistantMsg = { role: 'assistant', content: data.output };
      setMessages([...newMessages, assistantMsg]);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleClear = () => {
    setMessages([]);
    setError('');
  };

  if (!job && !error) return <div className="empty-state">Loading...</div>;

  return (
    <>
      <div className="page-title">
        <span>Chat</span>
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

      {/* Chat messages */}
      <div className="card" style={{ minHeight: '300px', maxHeight: '500px', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '12px', padding: '16px' }}>
        {messages.length === 0 && (
          <div style={{ color: '#666', fontSize: '13px', textAlign: 'center', margin: 'auto' }}>
            Send a message to start chatting with your fine-tuned model
          </div>
        )}
        {messages.map((msg, i) => (
          <div
            key={i}
            style={{
              alignSelf: msg.role === 'user' ? 'flex-end' : 'flex-start',
              background: msg.role === 'user' ? '#2563eb' : '#2a2a2a',
              color: '#e5e5e5',
              padding: '8px 12px',
              borderRadius: '12px',
              maxWidth: '80%',
              fontSize: '14px',
              lineHeight: '1.5',
              whiteSpace: 'pre-wrap',
            }}
          >
            {msg.content}
          </div>
        ))}
        {loading && (
          <div style={{ alignSelf: 'flex-start', color: '#888', fontSize: '13px', fontStyle: 'italic' }}>
            Generating...
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Error */}
      {error && (
        <div className="card" style={{ borderLeft: '4px solid #dc2626' }}>
          <div className="error-text">{error}</div>
        </div>
      )}

      {/* Input */}
      <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
        <div style={{ display: 'flex', gap: '8px' }}>
          <textarea
            placeholder={job?.status !== 'complete' ? 'Job must be complete to chat' : 'Type a message... (Enter to send)'}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={loading || job?.status !== 'complete'}
            style={{ flex: 1, minHeight: '44px', maxHeight: '120px', resize: 'vertical' }}
          />
          <button
            className="btn btn-primary"
            disabled={!input.trim() || loading || job?.status !== 'complete'}
            onClick={handleSend}
            style={{ alignSelf: 'flex-end' }}
          >
            Send
          </button>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', fontSize: '12px', color: '#888' }}>
          <span>Max tokens: {maxTokens}</span>
          <input
            type="range"
            min="10"
            max="200"
            value={maxTokens}
            onChange={(e) => setMaxTokens(Number(e.target.value))}
            style={{ flex: 1 }}
          />
          <button onClick={handleClear} style={{ fontSize: '12px', color: '#888', background: 'none', border: '1px solid #444', borderRadius: '4px', padding: '2px 8px', cursor: 'pointer' }}>
            Clear chat
          </button>
        </div>
      </div>
    </>
  );
}
