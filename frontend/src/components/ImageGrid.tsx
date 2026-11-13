import React, { useState, useCallback, useEffect } from 'react';
import { Images } from 'lucide-react';
import { CenteredSpinner, EmptyState, ErrorState } from './ui/States';
import { useQuery } from '@tanstack/react-query';
import { getImages } from '../hooks/useApi';
import ImageCard from './ImageCard';
import type { ImageSummary } from '../types';

interface ImageGridProps {
  folderId: number | null;
  selectedImageId: number | null;
  onSelectImage: (image: ImageSummary) => void;
}

const PAGE_SIZE = 50;

export default function ImageGrid({ folderId, selectedImageId, onSelectImage }: ImageGridProps) {
  const [offset, setOffset] = useState(0);

  // Reset offset when folder changes
  useEffect(() => setOffset(0), [folderId]);
  const queryFolderId = folderId;
  const { data, isLoading, isError, error, refetch } = useQuery({
    queryKey: ['images', queryFolderId, offset],
    queryFn: () =>
      getImages({
        limit: PAGE_SIZE,
        offset,
        ...(queryFolderId != null ? { folder_id: queryFolderId } : {}),
      }),
    refetchInterval: 10000,
  });

  const images = data?.images ?? [];
  const total = data?.total ?? 0;
  const totalPages = Math.ceil(total / PAGE_SIZE);
  const currentPage = Math.floor(offset / PAGE_SIZE) + 1;

  const goNext = useCallback(() => {
    if (offset + PAGE_SIZE < total) setOffset((o) => o + PAGE_SIZE);
  }, [offset, total]);

  const goPrev = useCallback(() => {
    setOffset((o) => Math.max(0, o - PAGE_SIZE));
  }, []);

  if (isLoading) return <CenteredSpinner />;

  if (isError) return <ErrorState error={error} onRetry={() => refetch()} />;

  if (images.length === 0) {
    return (
      <EmptyState icon={<Images className="h-6 w-6" />} title="No images indexed yet">
        Add a folder in the sidebar and press <b>Index</b>.
      </EmptyState>
    );
  }

  return (
    <div>
      <div className="grid grid-cols-[repeat(auto-fill,minmax(170px,1fr))] gap-3 px-5 py-4">
        {images.map((img) => (
          <ImageCard
            key={img.image_id}
            image={img}
            isSelected={img.image_id === selectedImageId}
            onClick={() => onSelectImage(img)}
          />
        ))}
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-center gap-3 border-t border-slate-800 px-4 py-3">
          <button
            onClick={goPrev}
            disabled={offset === 0}
            className="rounded-md px-3 py-1 text-xs text-slate-400 transition-colors hover:bg-slate-800 disabled:opacity-30"
          >
            ← Previous
          </button>
          <span className="text-xs text-slate-500">
            Page {currentPage} of {totalPages} ({total} images)
          </span>
          <button
            onClick={goNext}
            disabled={offset + PAGE_SIZE >= total}
            className="rounded-md px-3 py-1 text-xs text-slate-400 transition-colors hover:bg-slate-800 disabled:opacity-30"
          >
            Next →
          </button>
        </div>
      )}
    </div>
  );
}
