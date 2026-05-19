"use client";

import axios from "axios";
import { getApiUrl, switchToFallback } from "@/lib/api";

const api = axios.create({ timeout: 60000 });

api.interceptors.request.use(
  (config) => {
    config.baseURL = getApiUrl();
    if (!String(config.url || "").startsWith("http")) {
      config.url = config.baseURL + config.url;
      config.baseURL = "";
    }
    return config;
  },
  (error) => Promise.reject(error)
);

let backendType = null;
let lastBackendCheck = 0;

const detectBackendType = async () => {
  const now = Date.now();
  if (backendType && now - lastBackendCheck < 5 * 60 * 1000) return backendType;
  lastBackendCheck = now;
  const baseUrl = getApiUrl();
  try {
    const customResponse = await axios.get(`${baseUrl}/api/visiofirm/tools`, { timeout: 2000 });
    if (customResponse.status === 200) {
      backendType = "custom";
      return backendType;
    }
  } catch {
    try {
      const visiofirmResponse = await axios.get(`${baseUrl}/annotation/check_preannotation_status`, {
        params: { project_name: "test" },
        timeout: 2000,
      });
      if (visiofirmResponse.status === 200 && visiofirmResponse.data?.success !== undefined) {
        backendType = "visiofirm";
        return backendType;
      }
    } catch {}
  }
  return null;
};

const BUILTIN_MODELS_FALLBACK = [
  { id: "YOLO", name: "YOLOv8-nano", task: "detection", source: "builtin", isLocal: true },
  { id: "FasterRCNN", name: "Faster R-CNN", task: "detection", source: "builtin", isLocal: true },
  { id: "SSD", name: "SSD", task: "detection", source: "builtin", isLocal: true },
  { id: "ResNet", name: "ResNet50", task: "classification", source: "builtin", isLocal: true },
  { id: "EfficientNet", name: "EfficientNet", task: "classification", source: "builtin", isLocal: true },
  { id: "YOLO-Seg", name: "YOLOv8-Seg", task: "segmentation", source: "builtin", isLocal: true },
  { id: "MaskRCNN", name: "Mask R-CNN", task: "segmentation", source: "builtin", isLocal: true },
  { id: "SAM", name: "SAM", task: "segmentation", source: "builtin", isLocal: true },
];

function normalizeModelName(name, id) {
  const raw = String(name || id || "").trim();
  if (!raw) return String(id || "");
  return raw.replace(/\s*\(内置\)\s*/g, "").replace(/^SAM(\s+分割一切)?$/i, "SAM").trim();
}

export const visioFirmAPI = {
  getTools: async () => {
    try {
      const detectedBackend = await detectBackendType();
      if (detectedBackend === "custom") {
        return await api.get("/api/tools");
      }
      return { data: { tools: ["classification", "detection", "segmentation"] } };
    } catch {
      return { data: { tools: ["classification", "detection", "segmentation"] } };
    }
  },

  getModels: async () => {
    try {
      const response = await api.get("/api/visiofirm/models", { timeout: 10000 });
      if (Array.isArray(response.data) && response.data.length > 0) {
        return response.data.map((m) => ({ ...m, name: normalizeModelName(m.name, m.id) }));
      }
      const catalogResp = await api.get("/api/visiofirm/models/catalog", { timeout: 10000 });
      const catalog = Array.isArray(catalogResp.data) ? catalogResp.data : [];
      return [...BUILTIN_MODELS_FALLBACK, ...catalog.map((m) => ({ ...m, name: normalizeModelName(m.name, m.id) }))];
    } catch {
      return BUILTIN_MODELS_FALLBACK;
    }
  },

  uploadModel: async (file, name, task) => {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("name", name);
    formData.append("task", task);
    const response = await api.post("/api/visiofirm/models/upload", formData, {
      headers: { "Content-Type": "multipart/form-data" },
      timeout: 120000,
    });
    return response.data;
  },

  downloadModel: async (modelId) => {
    const formData = new FormData();
    formData.append("model_id", modelId);
    const response = await api.post("/api/visiofirm/models/download", formData, {
      headers: { "Content-Type": "multipart/form-data" },
      timeout: 300000,
    });
    return response.data;
  },

  autoAnnotate: async (formData, retryCount = 0) => {
    try {
      return await api.post("/api/visiofirm/annotate", formData, {
        headers: { "Content-Type": "multipart/form-data" },
        timeout: 60000,
      });
    } catch (error) {
      const shouldRetry = ["Network Error", "ERR_CONNECTION_REFUSED", "ECONNABORTED", "timeout of", "504", "503"].some(
        (s) => error?.message?.includes(s) || error?.response?.status === Number(s)
      );
      if (shouldRetry && retryCount < 3) {
        if (retryCount === 0) switchToFallback?.();
        const delay = Math.min(1000 * Math.pow(2, retryCount), 5000);
        await new Promise((r) => setTimeout(r, delay));
        return visioFirmAPI.autoAnnotate(formData, retryCount + 1);
      }
      throw error;
    }
  },
};

