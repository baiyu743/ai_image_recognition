"use client";

const ENV_CONFIG = {
  development: {
    API_BASE_URL: "http://localhost:8000",
    FALLBACK_API_URL: "http://114.55.52.100:8000",
    TIMEOUT: 30000,
  },
  production: {
    API_BASE_URL: "",
    FALLBACK_API_URL: "http://114.55.52.100:8000",
    TIMEOUT: 30000,
  },
};

const getCurrentEnv = () => {
  if (typeof window === "undefined") return "development";
  const hostname = window.location.hostname;
  if (hostname === "localhost" || hostname === "127.0.0.1" || hostname.startsWith("192.168.")) {
    return "development";
  }
  return "production";
};

const FORCE_ENV = "auto";

const getEnvConfig = () => {
  const env = FORCE_ENV === "auto" ? getCurrentEnv() : FORCE_ENV;
  const cfg = ENV_CONFIG[env] || ENV_CONFIG.development;
  if (env === "production") {
    const origin = typeof window !== "undefined" ? window.location.origin : "";
    return { ...cfg, API_BASE_URL: origin };
  }
  return cfg;
};

let currentApiUrl = null;
let usingFallback = false;

export const getApiUrl = () => {
  if (!currentApiUrl) {
    currentApiUrl = getEnvConfig().API_BASE_URL;
  }
  return currentApiUrl;
};

export const switchToFallback = () => {
  const env = FORCE_ENV === "auto" ? getCurrentEnv() : FORCE_ENV;
  if (env !== "production") {
    return false;
  }
  const cfg = getEnvConfig();
  if (cfg.FALLBACK_API_URL && !usingFallback) {
    currentApiUrl = cfg.FALLBACK_API_URL;
    usingFallback = true;
    return true;
  }
  return false;
};

export const resetToMainApi = () => {
  const cfg = getEnvConfig();
  currentApiUrl = cfg.API_BASE_URL;
  usingFallback = false;
};

export const TIMEOUT = getEnvConfig().TIMEOUT;

