import { useState, useEffect } from 'react';
import { getGpuStatus } from '../api';

export default function GpuBanner() {
  const [gpuOnline, setGpuOnline] = useState(null);

  useEffect(() => {
    const poll = async () => {
      try {
        const data = await getGpuStatus();
        setGpuOnline(data.gpu_pod === true);
      } catch {
        setGpuOnline(false);
      }
    };
    poll();
    const id = setInterval(poll, 10_000);
    return () => clearInterval(id);
  }, []);

  if (gpuOnline === null) return null;

  return (
    <div className={`gpu-banner ${gpuOnline ? 'gpu-banner--online' : 'gpu-banner--offline'}`}>
      <span className="gpu-banner-dot" />
      {gpuOnline
        ? 'GPU Pod Online \u2014 Submit jobs and chat with models'
        : 'GPU Pod Offline \u2014 Browse completed jobs. Training and inference unavailable.'}
    </div>
  );
}
