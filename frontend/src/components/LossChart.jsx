import { useEffect, useRef, useState } from 'react';
import { Chart as ChartJS, LinearScale, PointElement, LineElement, Tooltip, Legend } from 'chart.js';
import { Scatter } from 'react-chartjs-2';
import { streamJob } from '../api';

ChartJS.register(LinearScale, PointElement, LineElement, Tooltip, Legend);

export default function LossChart({ jobId, onStatusChange }) {
  const [trainData, setTrainData] = useState([]);
  const [evalData, setEvalData] = useState([]);
  const [status, setStatus] = useState('connecting');
  const currentStepRef = useRef(0);
  const streamRef = useRef(null);

  useEffect(() => {
    if (!jobId) return;

    setTrainData([]);
    setEvalData([]);
    setStatus('connecting');
    currentStepRef.current = 0;

    const es = streamJob(jobId);
    streamRef.current = es;

    es.onopen = () => {
      setStatus('streaming');
      onStatusChange?.('streaming');
    };

    es.onmessage = (event) => {
      const data = JSON.parse(event.data);

      if (data.type === 'step') {
        currentStepRef.current = data.step;
        setTrainData(prev => [...prev, { x: data.step, y: data.train_loss }]);
      } else if (data.type === 'eval') {
        setEvalData(prev => [...prev, { x: currentStepRef.current, y: data.eval_loss }]);
        setStatus(`epoch ${data.epoch} eval_loss=${data.eval_loss.toFixed(4)}`);
      } else if (data.type === 'done') {
        setStatus(`done (${data.status}), eval_loss=${data.eval_loss?.toFixed(4) ?? 'n/a'}`);
        onStatusChange?.('done');
        es.close();
        streamRef.current = null;
      }
    };

    es.onerror = () => {
      setStatus('connection error');
      onStatusChange?.('error');
      es.close();
      streamRef.current = null;
    };

    return () => {
      if (streamRef.current) {
        streamRef.current.close();
        streamRef.current = null;
      }
    };
  }, [jobId]);

  const statusClass = status.startsWith('done') ? 'done'
    : status === 'connection error' ? 'error'
    : status === 'connecting' ? 'idle'
    : 'streaming';

  const chartData = {
    datasets: [
      {
        label: 'train_loss',
        data: trainData,
        borderColor: '#2563eb',
        backgroundColor: 'transparent',
        pointRadius: 2,
        showLine: true,
        tension: 0.1,
      },
      {
        label: 'eval_loss',
        data: evalData,
        borderColor: '#dc2626',
        backgroundColor: '#dc2626',
        pointRadius: 5,
        showLine: false,
      },
    ],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      x: { type: 'linear', title: { display: true, text: 'step' } },
      y: { title: { display: true, text: 'loss' } },
    },
    animation: { duration: 0 },
  };

  return (
    <>
      <div className={`status-bar ${statusClass}`}>{status}</div>
      <div className="chart-container">
        <Scatter data={chartData} options={options} />
      </div>
    </>
  );
}
