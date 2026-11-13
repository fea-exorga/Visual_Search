import React from 'react';

interface SliderProps {
  label: React.ReactNode;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  format?: (v: number) => string;
  accent?: string;
  id?: string;
}

export default function Slider({ label, value, min, max, step, onChange, format, accent, id }: SliderProps) {
  const pct = ((value - min) / (max - min)) * 100;
  return (
    <label className="block" htmlFor={id}>
      <div className="mb-0.5 flex items-center justify-between text-xs">
        <span className="flex items-center gap-1.5 text-slate-400">
          {accent && <span className="h-2 w-2 rounded-full" style={{ backgroundColor: accent }} />}
          {label}
        </span>
        <span className="font-mono text-[11px] text-slate-300">{format ? format(value) : value}</span>
      </div>
      <input
        id={id}
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full"
        style={{ ['--pct' as string]: `${pct}%` } as React.CSSProperties}
      />
    </label>
  );
}
