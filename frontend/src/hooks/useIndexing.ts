import { useCallback, useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { getFolders, pauseIndex, resumeIndex, startIndex } from './useApi';
import { useSSE } from './useSSE';
import type { IndexProgress } from '../types';

export interface IndexingState {
  jobId: string | null;
  progress: IndexProgress | null;
  isRunning: boolean;
  isPaused: boolean;
  canStart: boolean;
  isStarting: boolean;
  error: string | null;
  start: (force?: boolean) => void;
  pause: () => void;
  resume: () => void;
}

/** App-wide indexing job state (one SSE connection regardless of the visible view). */
export function useIndexing(): IndexingState {
  const qc = useQueryClient();
  const [jobId, setJobId] = useState<string | null>(null);
  const [isPaused, setPaused] = useState(false);

  const { data: folders = [] } = useQuery({ queryKey: ['folders'], queryFn: getFolders, refetchInterval: 5000 });
  const enabled = folders.filter((f) => Boolean(f.enabled));

  const onDone = useCallback(() => {
    setJobId(null);
    setPaused(false);
    for (const k of ['folders', 'stats', 'images', 'galaxy', 'search', 'duplicates', 'systemInfo']) {
      qc.invalidateQueries({ queryKey: [k] });
    }
  }, [qc]);

  const { progress, isDone, error } = useSSE(jobId, onDone);

  const startM = useMutation({
    mutationFn: (force: boolean) => startIndex(enabled.map((f) => f.folder_id), force),
    onSuccess: (d) => {
      setPaused(false);
      setJobId(d.job_id);
      qc.invalidateQueries({ queryKey: ['folders'] });
    },
  });
  const pauseM = useMutation({ mutationFn: () => pauseIndex(jobId!), onSuccess: () => setPaused(true) });
  const resumeM = useMutation({ mutationFn: () => resumeIndex(jobId!), onSuccess: () => setPaused(false) });

  const startErr = startM.error as Error | null;
  return {
    jobId,
    progress,
    isRunning: jobId !== null && !isDone,
    isPaused,
    canStart: enabled.length > 0 && !startM.isPending,
    isStarting: startM.isPending,
    error: error ?? startErr?.message ?? null,
    start: (force = false) => startM.mutate(force),
    pause: () => jobId && pauseM.mutate(),
    resume: () => jobId && resumeM.mutate(),
  };
}
