import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [tailwindcss(), react()],
  test: {
    environment: "node",
    include: ["src/**/*.test.{js,jsx}", "mobile/**/*.test.{js,jsx}"],
  },
});
