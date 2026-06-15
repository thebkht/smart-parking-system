import axios from "axios";
import Constants from "expo-constants";

// In Expo Go the Metro bundler host is the dev machine — assume the backend
// runs there on :8000 unless EXPO_PUBLIC_API_BASE overrides it.
function deriveDevHostBase() {
  const host = Constants.expoConfig?.hostUri?.split(":")[0];
  return host ? `http://${host}:8000` : null;
}

export const API_BASE =
  process.env.EXPO_PUBLIC_API_BASE ??
  deriveDevHostBase() ??
  "http://127.0.0.1:8000";

export const api = axios.create({
  baseURL: API_BASE,
  timeout: 30000,
});

function requestUrl(config) {
  const baseURL = config?.baseURL ?? API_BASE;
  const url = config?.url ?? "";
  return url.startsWith("http") ? url : `${baseURL}${url}`;
}

function logError(error) {
  const config = error?.config ?? {};
  const response = error?.response;
  console.error("[api] request failed", {
    method: config.method?.toUpperCase(),
    url: requestUrl(config),
    timeout: config.timeout,
    status: response?.status,
    response: response?.data,
    message: error?.message,
    code: error?.code,
  });
}

// Optional bearer token — only needed when the backend runs with AUTH_ENABLED.
export const API_TOKEN = process.env.EXPO_PUBLIC_API_TOKEN ?? null;

api.interceptors.request.use((config) => {
  if (API_TOKEN) {
    config.headers = config.headers ?? {};
    config.headers.Authorization = `Bearer ${API_TOKEN}`;
  }
  console.log("[api] request", {
    method: config.method?.toUpperCase(),
    url: requestUrl(config),
    timeout: config.timeout,
    headers: config.headers,
  });
  return config;
});

api.interceptors.response.use(
  (response) => {
    console.log("[api] response", {
      method: response.config?.method?.toUpperCase(),
      url: requestUrl(response.config),
      status: response.status,
    });
    return response;
  },
  (error) => {
    logError(error);
    return Promise.reject(error);
  },
);
