import type { GalaxyResponse } from '../../types';

export interface FlyTarget {
  x: number;
  y: number;
  z?: number;
  /** Approximate radius (world units, coords in [-1,1]) that should fit the viewport. */
  spread: number;
  nonce: number;
}

export interface HoverInfo {
  imageId: number;
  clusterId: number;
  /** Position relative to the renderer container, CSS px. */
  sx: number;
  sy: number;
}

export interface GalaxyRendererProps {
  data: GalaxyResponse;
  clusterColors: Map<number, string>;
  /** Search hits — when non-null everything else is dimmed. */
  highlight: Set<number> | null;
  /** Cluster filter — when non-empty other clusters are dimmed and not hoverable. */
  activeClusters: Set<number>;
  focusId: number | null;
  fly: FlyTarget | null;
  onHover: (h: HoverInfo | null) => void;
  onSelect: (imageId: number) => void;
  autoRotate?: boolean;
}

export function easeInOutCubic(t: number) {
  return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
}
