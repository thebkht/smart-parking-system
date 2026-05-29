import { useState, useEffect, useRef, useCallback } from "react";
import {
  View,
  Text,
  TouchableOpacity,
  ScrollView,
  StyleSheet,
  ActivityIndicator,
} from "react-native";
import { api } from "../api";

const SPOT_COLORS = {
  free: "#1a7a4a",
  occupied: "#c0392b",
  unknown: "#e7e5e4",
};

export default function OccupancyMapScreen({ layout }) {
  const [status, setStatus] = useState(null);
  const [polling, setPolling] = useState(false);
  const [lastUpdated, setLastUpdated] = useState(null);
  const [selectedSpot, setSelectedSpot] = useState(null);
  const intervalRef = useRef(null);

  const poll = useCallback(async () => {
    try {
      const { data } = await api.get("/status");
      setStatus(data.spots ?? data);
      setLastUpdated(new Date());
    } catch {
      setLastUpdated(new Date());
    }
  }, []);

  const startPolling = () => {
    setPolling(true);
    poll();
    intervalRef.current = setInterval(poll, 3000);
  };

  const stopPolling = () => {
    clearInterval(intervalRef.current);
    setPolling(false);
  };

  useEffect(() => () => clearInterval(intervalRef.current), []);

  if (!layout) {
    return (
      <View style={styles.empty}>
        <Text style={styles.emptyText}>
          No layout loaded. Go to Owner Setup first.
        </Text>
      </View>
    );
  }

  const freeCount = status
    ? Object.values(status).filter((v) => v === "free").length
    : 0;
  const occCount = status
    ? Object.values(status).filter((v) => v === "occupied").length
    : 0;

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      {/* Stats */}
      <View style={styles.statsRow}>
        <View style={styles.statCard}>
          <Text style={styles.statLabel}>Free</Text>
          <Text style={[styles.statValue, { color: "#0f5c36" }]}>
            {status ? freeCount : "–"}
          </Text>
        </View>
        <View style={styles.statCard}>
          <Text style={styles.statLabel}>Occupied</Text>
          <Text style={[styles.statValue, { color: "#c0392b" }]}>
            {status ? occCount : "–"}
          </Text>
        </View>
        <View style={styles.statCard}>
          <Text style={styles.statLabel}>Total</Text>
          <Text style={[styles.statValue, { color: "#78716c" }]}>
            {layout.spots.length}
          </Text>
        </View>
      </View>

      {/* Controls */}
      <View style={styles.controls}>
        <TouchableOpacity
          style={[styles.btn, polling && styles.btnDisabled]}
          onPress={startPolling}
          disabled={polling}
        >
          <Text style={styles.btnText}>▶ Start polling</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[styles.btn, !polling && styles.btnDisabled]}
          onPress={stopPolling}
          disabled={!polling}
        >
          <Text style={styles.btnText}>■ Stop</Text>
        </TouchableOpacity>
        {polling && (
          <ActivityIndicator color="#78716c" style={{ marginLeft: 8 }} />
        )}
        {lastUpdated && (
          <Text style={styles.timestamp}>
            {lastUpdated.toLocaleTimeString()}
          </Text>
        )}
      </View>

      {/* Spot grid */}
      <View style={styles.grid}>
        {layout.spots.map((spot) => {
          const occ = status?.[spot.spot_id] ?? "unknown";
          const isSelected = selectedSpot === spot.spot_id;
          return (
            <TouchableOpacity
              key={spot.spot_id}
              style={[
                styles.spotCard,
                { backgroundColor: SPOT_COLORS[occ] },
                isSelected && styles.spotSelected,
              ]}
              onPress={() => setSelectedSpot(isSelected ? null : spot.spot_id)}
            >
              <Text style={styles.spotLabel}>
                {spot.spot_id.replace("spot_", "P")}
              </Text>
              <Text style={styles.spotStatus}>{occ}</Text>
            </TouchableOpacity>
          );
        })}
      </View>

      {/* Selected spot */}
      {selectedSpot && status && (
        <View style={styles.selectedPanel}>
          <Text style={styles.selectedId}>{selectedSpot}</Text>
          <Text
            style={[
              styles.selectedStatus,
              {
                color: status[selectedSpot] === "free" ? "#0f5c36" : "#c0392b",
              },
            ]}
          >
            {status[selectedSpot] ?? "unknown"}
          </Text>
          <TouchableOpacity onPress={() => setSelectedSpot(null)}>
            <Text style={styles.dismiss}>✕</Text>
          </TouchableOpacity>
        </View>
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#fafaf9" },
  content: { padding: 20 },
  empty: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    padding: 40,
  },
  emptyText: { fontSize: 15, color: "#78716c", textAlign: "center" },
  statsRow: { flexDirection: "row", gap: 10, marginBottom: 20 },
  statCard: {
    flex: 1,
    backgroundColor: "#f5f5f4",
    borderRadius: 12,
    padding: 14,
    alignItems: "center",
  },
  statLabel: {
    fontSize: 11,
    color: "#a8a29e",
    textTransform: "uppercase",
    marginBottom: 4,
  },
  statValue: { fontSize: 32, fontWeight: "500", fontVariant: ["tabular-nums"] },
  controls: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
    marginBottom: 20,
  },
  btn: {
    borderWidth: 1,
    borderColor: "#d4d0cb",
    borderRadius: 8,
    paddingHorizontal: 14,
    paddingVertical: 8,
  },
  btnDisabled: { opacity: 0.4 },
  btnText: { fontSize: 13, color: "#44403c" },
  timestamp: {
    fontSize: 12,
    color: "#a8a29e",
    marginLeft: "auto",
    fontVariant: ["tabular-nums"],
  },
  grid: { flexDirection: "row", flexWrap: "wrap", gap: 8 },
  spotCard: {
    width: "30%",
    borderRadius: 10,
    padding: 12,
    alignItems: "center",
    marginBottom: 4,
  },
  spotSelected: { borderWidth: 2, borderColor: "#1c1917" },
  spotLabel: {
    fontSize: 14,
    fontWeight: "600",
    color: "#fff",
    marginBottom: 2,
  },
  spotStatus: { fontSize: 11, color: "rgba(255,255,255,0.8)" },
  selectedPanel: {
    marginTop: 16,
    flexDirection: "row",
    alignItems: "center",
    padding: 14,
    backgroundColor: "#f5f5f4",
    borderRadius: 12,
    gap: 12,
  },
  selectedId: { fontSize: 15, fontWeight: "600", color: "#1c1917", flex: 1 },
  selectedStatus: { fontSize: 14, fontWeight: "500" },
  dismiss: { fontSize: 16, color: "#a8a29e" },
});
