"use client";

const STORAGE_KEY = "ai_model_current_project";

export function getCurrentProject() {
  if (typeof window === "undefined") return null;
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (!parsed || !parsed.id) return null;
    return { id: parsed.id, name: parsed.name || "" };
  } catch {
    return null;
  }
}

export function setCurrentProject(project) {
  if (typeof window === "undefined") return;
  const payload = { id: project?.id || "", name: project?.name || "" };
  if (!payload.id) return;
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
}

export function clearCurrentProject() {
  if (typeof window === "undefined") return;
  window.localStorage.removeItem(STORAGE_KEY);
}

