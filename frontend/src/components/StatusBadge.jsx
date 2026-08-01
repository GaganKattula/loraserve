const CLASS_MAP = {
  queued: 'badge-queued',
  running: 'badge-running',
  complete: 'badge-complete',
  failed: 'badge-failed',
};

export default function StatusBadge({ status }) {
  return <span className={`badge ${CLASS_MAP[status] || ''}`}>{status}</span>;
}
