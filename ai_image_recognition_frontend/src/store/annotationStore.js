"use client";

import { create } from "zustand";

export const useAnnotationStore = create((set, get) => ({
  uploadedImages: [],
  currentImageIndex: -1,
  annotationMode: "manual",
  selectedTool: "object_detection",
  selectedModel: "",
  exportFormat: "json",
  galleryFilter: "all",

  imageAnnotations: {},
  savedAnnotations: {},

  setState: (patch) => set(patch),

  reset: () =>
    set({
      uploadedImages: [],
      currentImageIndex: -1,
      annotationMode: "manual",
      selectedTool: "object_detection",
      selectedModel: "",
      exportFormat: "json",
      galleryFilter: "all",
      imageAnnotations: {},
      savedAnnotations: {},
    }),
}));

