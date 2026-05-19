"use client";

const STORAGE_KEY = "ai_model_user_config";

const DEFAULT_CONFIG = {
  lastExportPath: "",
  augmentation: {
    providerPreset: "",
    apiStyle: "",
    baseUrl: "",
    imageUrl: "",
    textModel: "",
    imageModel: "",
  },
};

export function getUserConfig() {
  if (typeof window === "undefined") return DEFAULT_CONFIG;
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return DEFAULT_CONFIG;
    const parsed = JSON.parse(raw);
    return { ...DEFAULT_CONFIG, ...(parsed || {}) };
  } catch {
    return DEFAULT_CONFIG;
  }
}

export function saveUserConfig(cfg) {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(cfg || DEFAULT_CONFIG));
}

