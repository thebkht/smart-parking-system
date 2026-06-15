# SmartParking — Frontend

Web app (Vite + React) and mobile app (Expo + React Native) for the SmartParking v6 system.

**Platform split (final):**
- **Web** — Owner Setup + Live Occupancy Map
- **Mobile** — Live Occupancy Map + Find My Car

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

Copy `.env.example` to `.env.local` and set `VITE_API_BASE`:

```bash
cp .env.example .env.local
# edit .env.local
```

```env
VITE_API_BASE=http://localhost:8000
```

For LAN testing (phone or other device on the same Wi-Fi), use the host machine's IP:

```env
VITE_API_BASE=http://192.168.1.100:8000
```

### Run

```bash
npm run dev
```

### Screens

**Owner Setup** — upload 4–5 overlapping parking lot photos, submit to `POST /layout` (which runs SfM server-side), and preview the generated BEV map with spot polygon overlays. After generation, a **label-correction step** lets the owner rename each spot (persisted via `PATCH /spots/{id}`); edited labels appear on the map. If SfM cannot build a layout the backend returns `422` and the UI prompts for manual polygon submission (`POST /map`) or loading the sample handoff. When the backend runs with `AUTH_ENABLED`, an optional owner sign-in widget obtains a bearer token (stored in `localStorage` as `spp_token`) that is attached to mutating requests.

**Live Occupancy Map** — polls `GET /status` every 3 seconds and colors each spot polygon green (free) or red (occupied) on a Leaflet map. Shows free/occupied/total counts in the header.

### Tech stack

- Vite + React
- Leaflet for polygon map rendering (raw `leaflet`, no `react-leaflet`)
- Tailwind CSS v4
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

Copy `.env.example` to `.env.local` and set `EXPO_PUBLIC_API_BASE` if needed:

```bash
cp .env.example .env.local
```

In Expo Go the backend host is **auto-detected** from the Metro bundler IP (`Constants.expoConfig.hostUri`), so you usually don't need this. Set it explicitly only if auto-detection fails or you're pointing at a different machine.

Do **not** use `127.0.0.1` or `localhost` on a physical device — those resolve to the phone itself.

If the backend runs with `AUTH_ENABLED`, set `EXPO_PUBLIC_API_TOKEN` to a bearer
token (from `POST /auth/register`) and it will be attached to all requests.
Otherwise leave it unset — the demo runs token-free.

### Run

```bash
npx expo start
```

Scan the QR code with the Expo Go app, or press `a` for Android emulator. Keep the backend terminal open (`make backend`) before testing from a phone.

### Screens

**Live Occupancy Map** — fetches `GET /map` on launch, polls `GET /status` every 3 seconds, renders spot quadrilaterals at exact coordinates with pinch-to-zoom and pan. Counts Free / Occupied / Total scoped to the active layout.

**Find My Car** — take or upload a photo, POST to `/park` to run SIFT localization, store the returned `session_id`, then GET `/find/{session_id}` to highlight your spot in amber on the coordinate-accurate map.

### Tech stack

- Expo SDK 54
- React Navigation (bottom tabs)
- react-native-svg for coordinate-accurate polygon map
- react-native-gesture-handler + react-native-reanimated for pinch/pan
- expo-image-picker for photo selection
- Axios for API calls

---

## API Contract

| Method | Path                 | Used by                              |
| ------ | -------------------- | ------------------------------------ |
| `POST` | `/layout`            | Web — Owner Setup                    |
| `GET`  | `/map`               | Both — fetch lot layout              |
| `GET`  | `/map/background`    | Both — BEV background image          |
| `GET`  | `/status`            | Both — Live Occupancy polling        |
| `POST` | `/park`              | Mobile — Find My Car (park photo)    |
| `GET`  | `/find/{session_id}` | Mobile — Find My Car (locate spot)   |
