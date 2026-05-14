import { BrowserRouter, Routes, Route, Link } from "react-router-dom";
import "leaflet/dist/leaflet.css";
import { MapContainer, TileLayer, Polygon } from "react-leaflet";

const spots = [
  {
    id: "spot_1",
    status: "free",
    coords: [
      [51.505, -0.09],
      [51.506, -0.09],
      [51.506, -0.08],
      [51.505, -0.08],
    ],
  },
  {
    id: "spot_2",
    status: "occupied",
    coords: [
      [51.507, -0.09],
      [51.508, -0.09],
      [51.508, -0.08],
      [51.507, -0.08],
    ],
  },
];

function OwnerSetup() {
  return <h1>Owner Setup</h1>;
}

function OccupancyMap() {
  return (
    <div>
      <h1>Live Occupancy Map</h1>
      <MapContainer
        center={[51.505, -0.09]}
        zoom={17}
        style={{ height: "500px", width: "100%" }}
      >
        <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
        {spots.map((spot) => (
          <Polygon
            key={spot.id}
            positions={spot.coords}
            pathOptions={{ color: spot.status === "free" ? "green" : "red" }}
          />
        ))}
      </MapContainer>
    </div>
  );
}

function FindMyCar() {
  return <h1>Find My Car</h1>;
}

function App() {
  return (
    <BrowserRouter>
      <nav>
        <Link to="/">Owner Setup</Link> | <Link to="/map">Occupancy Map</Link> |{" "}
        <Link to="/find">Find My Car</Link>
      </nav>
      <Routes>
        <Route path="/" element={<OwnerSetup />} />
        <Route path="/map" element={<OccupancyMap />} />
        <Route path="/find" element={<FindMyCar />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
