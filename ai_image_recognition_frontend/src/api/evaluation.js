"use client";

import axios from "axios";
import { TIMEOUT, getApiUrl, switchToFallback } from "@/lib/api";

const api = axios.create({
  baseURL: getApiUrl(),
  timeout: TIMEOUT,
  headers: { "Content-Type": "application/json" },
});

api.interceptors.request.use(
  (config) => {
    config.baseURL = getApiUrl();
    return config;
  },
  (error) => Promise.reject(error)
);

api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error?.message?.includes("Network Error") || error?.message?.includes("ERR_CONNECTION_REFUSED")) {
      switchToFallback?.();
    }
    return Promise.reject(error);
  }
);

export const evaluationAPI = {
  startEvaluation: (modelId, data) => api.post(`/api/models/${modelId}/evaluate`, data),
  getEvaluationResult: (evaluationId, modelId = "yolov8n") => api.get(`/api/models/${modelId}/evaluation/${evaluationId}`),
  getLatestEvaluationData: (projectId) => api.get(`/api/projects/${projectId}/artifacts/evaluation-data`),
  listProjectEvaluations: (projectId, limit = 50) => api.get(`/api/projects/${projectId}/evaluations`, { params: { limit } }),
  getProjectEvaluation: (projectId, evaluationId) => api.get(`/api/projects/${projectId}/evaluations/${evaluationId}`),
  regenerateLatestEvaluationData: (projectId) => api.post(`/api/projects/${projectId}/artifacts/evaluation-data/regenerate`),
};

