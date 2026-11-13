import React from 'react';
import { AlertTriangle, PlugZap, RefreshCw, Construction } from 'lucide-react';
import { describeError } from '../../hooks/useApi';

export function Spinner({ className = 'h-6 w-6' }: { className?: string }) {
  return (
    <div
      role="status"
      aria-label="Loading"
      className={`${className} animate-spin rounded-full border-2 border-slate-700 border-t-indigo-400`}
    />
  );
}

export function CenteredSpinner({ label }: { label?: string }) {
  return (
    <div className="flex flex-1 flex-col items-center justify-center gap-3 py-20 text-sm text-slate-400">
      <Spinner className="h-8 w-8" />
      {label && <span>{label}</span>}
    </div>
  );
}

interface EmptyStateProps {
  icon?: React.ReactNode;
  title: string;
  children?: React.ReactNode;
  action?: React.ReactNode;
}

export function EmptyState({ icon, title, children, action }: EmptyStateProps) {
  return (
    <div className="flex flex-1 flex-col items-center justify-center px-6 py-16 text-center">
      {icon && (
        <div className="mb-4 flex h-14 w-14 items-center justify-center rounded-2xl border border-indigo-500/20 bg-indigo-500/10 text-indigo-300 shadow-[0_0_40px_-8px] shadow-indigo-500/40">
          {icon}
        </div>
      )}
      <h3 className="text-base font-semibold text-slate-200">{title}</h3>
      {children && <div className="mt-1.5 max-w-md text-sm text-slate-400">{children}</div>}
      {action && <div className="mt-5">{action}</div>}
    </div>
  );
}

export function ErrorState({ error, onRetry, compact }: { error: unknown; onRetry?: () => void; compact?: boolean }) {
  const d = describeError(error);
  const Icon = d.kind === 'down' ? PlugZap : d.kind === 'missing' ? Construction : AlertTriangle;
  if (compact) {
    return (
      <div className="flex items-start gap-2 rounded-lg border border-rose-500/20 bg-rose-500/5 px-3 py-2 text-xs text-rose-300">
        <Icon className="mt-0.5 h-3.5 w-3.5 shrink-0" />
        <div className="min-w-0">
          <span className="font-medium">{d.title}.</span> <span className="text-rose-300/80">{d.detail}</span>
        </div>
      </div>
    );
  }
  return (
    <EmptyState
      icon={<Icon className="h-6 w-6 text-rose-300" />}
      title={d.title}
      action={
        onRetry && (
          <button onClick={onRetry} className="btn-secondary">
            <RefreshCw className="h-3.5 w-3.5" /> Retry
          </button>
        )
      }
    >
      {d.detail}
    </EmptyState>
  );
}
