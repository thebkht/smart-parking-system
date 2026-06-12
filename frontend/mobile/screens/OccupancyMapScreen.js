import { useState, useEffect, useRef, useCallback } from "react";
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  ActivityIndicator,
} from "react-native";
import { api } from "../api";
import LotMap from "../components/LotMap";

export default function OccupancyMapScreen({ layout, layoutError, onRetry }) {
  const [status, setStatus] = useState(null);
  const [polling, setPolling] = useState(false);
  const [lastUpdated, setLastUpdated] = useState(null);
  const [selectedSpot, setSelectedSpot] = useState(null);
  const intervalRef = useRef(null);

  const poll = useCallback(async () => {
    try {
      const { data } = await api.get("/status");
      setStatus(data.spots ?? data);
      console.log("[OccupancyMap] status loaded", {
        statusCount: Object.keys(data.spots ?? data ?? {}).length,
        layoutCount: layout?.spots?.length ?? 0,
      });
      setLastUpdated(new Date());
    } catch (error) {
      console.warn("[OccupancyMap] status poll failed", {
        message: error?.message,
        status: error?.response?.status,
        response: error?.response?.data,
      });
      setLastUpdated(new Date());
    }
  }, [layout?.spots?.length]);

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
          Lot layout not available — publish it from the web dashboard.
        </Text>
        {layoutError && <Text style={styles.emptyDetail}>{layoutError}</Text>}
        {onRetry && (
          <TouchableOpacity style={styles.retryBtn} onPress={onRetry}>
            <Text style={styles.retryText}>Retry</Text>
          </TouchableOpacity>
        )}
      </View>
    );
  }

  const layoutStatuses = layout.spots.map(
    (spot) => status?.[spot.spot_id] ?? "unknown",
  );
  const freeCount = status
    ? layoutStatuses.filter((value) => value === "free").length
    : 0;
  const occCount = status
    ? layoutStatuses.filter((value) => value === "occupied").length
    : 0;

  return (
    <View style={[styles.container, styles.content]}>
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

      {/* Coordinate-accurate lot map — pinch to zoom, drag to pan */}
      <LotMap
        layout={layout}
        status={status}
        selectedSpot={selectedSpot}
        onSpotPress={(spotId) =>
          setSelectedSpot(selectedSpot === spotId ? null : spotId)
        }
      />
      <Text style={styles.mapHint}>
        pinch to zoom · drag to pan · tap a spot to select
      </Text>

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
    </View>
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
  emptyDetail: {
    fontSize: 13,
    color: "#a8a29e",
    textAlign: "center",
    marginTop: 8,
  },
  retryBtn: {
    marginTop: 16,
    borderWidth: 1,
    borderColor: "#d4d0cb",
    borderRadius: 8,
    paddingHorizontal: 20,
    paddingVertical: 10,
  },
  retryText: { fontSize: 14, color: "#44403c" },
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
  mapHint: {
    fontSize: 12,
    color: "#a8a29e",
    textAlign: "center",
    marginTop: 8,
  },
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
