# SmartParking — Frontend

Web app (Vite + React) and mobile app (Expo + React Native) for the SmartParking v6 system.

---

## Structure

```
frontend/          — web app (Vite + React + Leaflet)
frontend/mobile/   — mobile app (Expo + React Native)
```

---

## Web App

### Setup

```bash
cd frontend
npm install
```

### Environment

The API base URL is hardcoded in `src/App.jsx`:

```js
const API_BASE = "http://172.16.32.43:8000";
```

Change this to match your backend address before running.

### Run

```bash
npm run dev
```

### Screens

**Owner Setup** — upload 4–5 overlapping parking lot photos, submit to `POST /layout`, and preview the generated BEV map with spot polygon overlays. Falls back to `GET /map` if SfM is not wired end-to-end.

**Live Occupancy Map** — polls `GET /status` every 3 seconds and colors each spot polygon green (free) or red (occupied) on a Leaflet map. Shows free/occupied/total counts in the header.

**Find My Car** — take or upload a photo, POST to `/park` to run SIFT localization, store the returned `session_id`, then GET `/find/{session_id}` to highlight your spot in amber on the map.

### Tech stack

- Vite + React
- Leaflet for polygon map rendering
- Tailwind CSS v4 with custom stone/green/red/amber theme tokens
- Radix UI icons
- Axios for all API calls

---

## Mobile App

### Setup

```bash
cd frontend/mobile
npm install
```

### Environment

The API base URL is in `frontend/mobile/api.js`:

```js
const API_BASE = "http://172.16.32.43:8000";
```

Change this to match your backend address.

### Run

```bash
npx expo start
```

Scan the QR code with the Expo Go app on your phone, or press `a` for Android emulator.

### Screens

Same three screens as the web app — Owner Setup, Live Occupancy, Find My Car — built with native React Native components and bottom tab navigation.

### Tech stack

- Expo SDK 56
- React Navigation (bottom tabs)
- expo-camera + expo-image-picker for native camera access
- Axios for API calls

---

## API Contract

The frontend expects the backend at `http://172.16.32.43:8000` with these endpoints:

| Method | Path                 | Used by                                                     |
| ------ | -------------------- | ----------------------------------------------------------- |
| `POST` | `/layout`            | Owner Setup                                                 |
| `GET`  | `/map`               | Owner Setup fallback                                        |
| `GET`  | `/status`            | Live Occupancy — returns `{ spots, confidence, timestamp }` |
| `POST` | `/park`              | Find My Car                                                 |
| `GET`  | `/find/{session_id}` | Find My Car                                                 |
