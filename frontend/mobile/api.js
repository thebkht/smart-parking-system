import axios from "axios";

const API_BASE = "http://172.16.32.43:8000";

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

api.interceptors.request.use((config) => {
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
