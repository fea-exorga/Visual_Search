import { useState, useEffect, useRef } from 'react';
import type { IndexProgress } from '../types';

const BASE = '/api/v1';
const MAX_RETRIES = 5;

interface UseSSEReturn {
  progress: IndexProgress | null;
  isConnected: boolean;
  isDone: boolean;
  error: string | null;
}

/**
 * Subscribes to GET /index/stream for a job. The backend emits `event: done`
 * when the job finishes; after that we stop (no reconnect loop).
 */
export function useSSE(jobId: string | null, onDone?: () => void): UseSSEReturn {
  const [progress, setProgress] = useState<IndexProgress | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [isDone, setIsDone] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const onDoneRef = useRef(onDone);
  onDoneRef.current = onDone;

  useEffect(() => {
    setProgress(null);
    setIsConnected(false);
    setIsDone(false);
    setError(null);
    if (!jobId) return;

    let es: EventSource | null = null;
    let retries = 0;
    let timer: number | undefined;
    let closed = false;

    const connect = () => {
      if (closed) return;
      es = new EventSource(`${BASE}/index/stream?job_id=${encodeURIComponent(jobId)}`);
      es.onopen = () => {
        setIsConnected(true);
        setError(null);
        retries = 0;
      };
      es.onmessage = (event) => {
        try {
          setProgress(JSON.parse(event.data) as IndexProgress);
        } catch {
          /* ignore malformed frames */
        }
      };
      es.addEventListener('done', () => {
        closed = true;
        es?.close();
        setIsConnected(false);
        setIsDone(true);
        onDoneRef.current?.();
      });
      es.onerror = () => {
        es?.close();
        setIsConnected(false);
        if (closed) return;
        retries += 1;
        if (retries > MAX_RETRIES) {
          setError('Lost connection to the indexing stream');
          setIsDone(true);
          onDoneRef.current?.();
          return;
        }
        timer = window.setTimeout(connect, Math.min(1000 * 2 ** retries, 15000));
      };
    };

    connect();
    return () => {
      closed = true;
      window.clearTimeout(timer);
      es?.close();
    };
  }, [jobId]);

  return { progress, isConnected, isDone, error };
}
