import axios from "axios";

const API_BASE = "http://172.16.32.43:8000";

export const api = axios.create({
  baseURL: API_BASE,
  timeout: 30000,
});
