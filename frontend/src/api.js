const API = '';

export async function listJobs(limit = 20, offset = 0) {
  const res = await fetch(`${API}/jobs?limit=${limit}&offset=${offset}`);
  if (!res.ok) throw new Error(`Failed to list jobs: ${res.status}`);
  return res.json();
}

export async function getJob(id) {
  const res = await fetch(`${API}/jobs/${id}`);
  if (!res.ok) throw new Error(`Failed to get job: ${res.status}`);
  return res.json();
}

export async function createJob(taskDescription, examples) {
  const res = await fetch(`${API}/jobs`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ task_description: taskDescription, examples }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Failed to create job: ${res.status}`);
  }
  return res.json();
}

export async function runInference(jobId, text, maxNewTokens = 20) {
  const res = await fetch(`${API}/infer/${jobId}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, max_new_tokens: maxNewTokens }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Inference failed: ${res.status}`);
  }
  return res.json();
}

export function streamJob(jobId) {
  return new EventSource(`${API}/jobs/${jobId}/stream`);
}
