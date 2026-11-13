import React, { useState } from 'react';
import { Plus, X } from 'lucide-react';
import { contrastText } from '../../lib/format';
import { MAX_COLORS } from './searchState';

const PRESETS = [
  '#E53935', '#FB8C00', '#FDD835', '#7CB342', '#43A047', '#00897B', '#00ACC1', '#1E88E5',
  '#3949AB', '#8E24AA', '#D81B60', '#6D4C41', '#000000', '#757575', '#FFFFFF', '#F5DEB3',
];

interface ColorPickerProps {
  colors: string[];
  onChange: (colors: string[]) => void;
}

export default function ColorPicker({ colors, onChange }: ColorPickerProps) {
  const [custom, setCustom] = useState('#7C3AED');
  const full = colors.length >= MAX_COLORS;
  const add = (c: string) => {
    const u = c.toUpperCase();
    if (colors.includes(u)) onChange(colors.filter((x) => x !== u));
    else if (!full) onChange([...colors, u]);
  };

  return (
    <div>
      <div className="field-label">
        <span>Colour palette</span>
        <span className="font-mono normal-case tracking-normal text-slate-600">
          {colors.length}/{MAX_COLORS}
        </span>
      </div>
      <div className="grid grid-cols-8 gap-1.5">
        {PRESETS.map((c) => {
          const active = colors.includes(c.toUpperCase());
          return (
            <button
              key={c}
              onClick={() => add(c)}
              disabled={!active && full}
              aria-pressed={active}
              aria-label={`${active ? 'Remove' : 'Add'} colour ${c}`}
              title={c}
              className={`aspect-square rounded-md ring-offset-2 ring-offset-slate-900 transition hover:scale-110 disabled:cursor-not-allowed disabled:opacity-30 disabled:hover:scale-100 ${
                active ? 'ring-2 ring-indigo-300' : 'ring-1 ring-white/10'
              }`}
              style={{ backgroundColor: c }}
            />
          );
        })}
      </div>
      <div className="mt-2 flex items-center gap-2">
        <input
          type="color"
          value={custom}
          onChange={(e) => setCustom(e.target.value)}
          className="h-7 w-9 rounded-md ring-1 ring-white/10"
          aria-label="Pick a custom colour"
        />
        <button onClick={() => add(custom)} disabled={full} className="btn-secondary h-7 flex-1">
          <Plus className="h-3 w-3" /> Add {custom.toUpperCase()}
        </button>
      </div>
      {colors.length > 0 && (
        <div className="mt-2 flex flex-wrap gap-1.5">
          {colors.map((c) => (
            <button
              key={c}
              onClick={() => onChange(colors.filter((x) => x !== c))}
              className="chip border-transparent font-mono"
              style={{ backgroundColor: c, color: contrastText(c) }}
              aria-label={`Remove colour ${c}`}
            >
              {c}
              <X className="h-3 w-3" />
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
