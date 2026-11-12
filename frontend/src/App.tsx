import React, { useState } from 'react';
import Sidebar from './components/Sidebar';
import CenterPanel from './components/CenterPanel';
import InspectorDrawer from './components/InspectorDrawer';
import type { ImageSummary } from './types';

export default function App() {
  const [selectedFolderId, setSelectedFolderId] = useState<number | null>(null);
  const [selectedImage, setSelectedImage] = useState<ImageSummary | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-slate-950 text-slate-100">
      {/* Left Sidebar — 260px fixed */}
      <Sidebar
        selectedFolderId={selectedFolderId}
        onSelectFolder={setSelectedFolderId}
      />

      {/* Center — flexible */}
      <CenterPanel
        folderId={selectedFolderId}
        selectedImageId={selectedImage?.image_id ?? null}
        onSelectImage={(img) => setSelectedImage(img)}
        jobId={jobId}
        onJobStarted={(id) => setJobId(id)}
      />

      {/* Right Inspector Drawer — 340px, slides in */}
      <InspectorDrawer
        imageId={selectedImage?.image_id ?? null}
        onClose={() => setSelectedImage(null)}
      />
    </div>
  );
}
