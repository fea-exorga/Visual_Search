import { useState, useEffect, useRef, useCallback } from 'react';
import type { IndexProgress } from '../types';

const BASE = '/api/v1';

interface UseSSEReturn {
  progress: IndexProgress | null;
  isConnected: boolean;
  error: string | null;
}

export function useSSE(jobId: string | null): UseSSEReturn {
  const [progress, setProgress] = useState<IndexProgress | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const esRef = useRef<EventSource | null>(null);
  const retriesRef = useRef(0);

  const connect = useCallback(() => {
    if (!jobId) return;

    const url = `${BASE}/index/stream?job_id=${encodeURIComponent(jobId)}`;
    const es = new EventSource(url);
    esRef.current = es;

    es.onopen = () => {
      setIsConnected(true);
      setError(null);
      retriesRef.current = 0;
    };

    es.onmessage = (event) => {
      try {
        const data: IndexProgress = JSON.parse(event.data);
        setProgress(data);
      } catch {
        // ignore parse errors
      }
    };

    es.onerror = () => {
      es.close();
      setIsConnected(false);
      retriesRef.current += 1;
      const delay = Math.min(1000 * 2 ** retriesRef.current, 30000);
      setTimeout(connect, delay);
    };
  }, [jobId]);

  useEffect(() => {
    if (!jobId) {
      setProgress(null);
      setIsConnected(false);
      return;
    }

    connect();

    return () => {
      esRef.current?.close();
      esRef.current = null;
      setIsConnected(false);
    };
  }, [jobId, connect]);

  return { progress, isConnected, error };
}
