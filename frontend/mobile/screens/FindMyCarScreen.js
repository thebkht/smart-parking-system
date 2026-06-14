import { useState } from "react";
import {
  View,
  Text,
  TouchableOpacity,
  ActivityIndicator,
  ScrollView,
  StyleSheet,
  Image,
  Alert,
} from "react-native";
import * as ImagePicker from "expo-image-picker";
import { api } from "../api";
import { parseParkResponse, parseFindResponse } from "../findMyCar";
import LotMap from "../components/LotMap";

export default function FindMyCarScreen({ layout, layoutError, onRetry }) {
  const [step, setStep] = useState("idle");
  const [photo, setPhoto] = useState(null);
  const [sessionId, setSessionId] = useState(null);
  const [foundSpot, setFoundSpot] = useState(null);
  const [confidence, setConfidence] = useState(null);
  const [similarityScore, setSimilarityScore] = useState(null);
  const [error, setError] = useState(null);

  const takePhoto = async () => {
    const perm = await ImagePicker.requestCameraPermissionsAsync();
    if (!perm.granted) {
      Alert.alert("Permission required", "Camera access is needed.");
      return;
    }
    const result = await ImagePicker.launchCameraAsync({ quality: 0.8 });
    if (!result.canceled) {
      setPhoto(result.assets[0]);
      setStep("ready");
      setFoundSpot(null);
      setSessionId(null);
      setConfidence(null);
      setSimilarityScore(null);
      setError(null);
    }
  };

  const pickPhoto = async () => {
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ["images"],
      quality: 0.8,
    });
    if (!result.canceled) {
      setPhoto(result.assets[0]);
      setStep("ready");
      setFoundSpot(null);
      setSessionId(null);
      setConfidence(null);
      setSimilarityScore(null);
      setError(null);
    }
  };

  const park = async () => {
    setStep("matching");
    setError(null);
    const form = new FormData();
    form.append("photo", {
      uri: photo.uri,
      name: "photo.jpg",
      type: "image/jpeg",
    });
    try {
      const { data } = await api.post("/park", form, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setSessionId(parseParkResponse(data).sessionId);
      setStep("parked");
    } catch (err) {
      setError(err?.response?.data?.detail ?? "Could not reach backend.");
      setStep("ready");
    }
  };

  const find = async () => {
    setStep("locating");
    try {
      const { data } = await api.get(`/find/${sessionId}`);
      const parsed = parseFindResponse(data);
      setFoundSpot(parsed.spotId);
      setConfidence(parsed.confidence);
      setSimilarityScore(parsed.similarityScore);
      setStep("found");
    } catch (err) {
      setError(err?.response?.data?.detail ?? "Could not locate your car.");
      setStep("parked");
    }
  };

  const reset = () => {
    setStep("idle");
    setPhoto(null);
    setSessionId(null);
    setFoundSpot(null);
    setConfidence(null);
    setSimilarityScore(null);
    setError(null);
  };

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

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      <Text style={styles.title}>Find My Car</Text>
      <Text style={styles.subtitle}>
        Take a photo from near where you parked. SIFT feature matching will
        identify your spot.
      </Text>

      {/* Photo preview */}
      {photo ? (
        <Image source={{ uri: photo.uri }} style={styles.preview} />
      ) : (
        <View style={styles.photoPlaceholder}>
          <Text style={styles.photoPlaceholderText}>No photo yet</Text>
        </View>
      )}

      {/* Error */}
      {error && <Text style={styles.error}>{error}</Text>}

      {/* Action buttons */}
      <View style={styles.buttonGroup}>
        {(step === "idle" || step === "ready") && (
          <>
            <TouchableOpacity style={styles.btnPrimary} onPress={takePhoto}>
              <Text style={styles.btnPrimaryText}>📷 Take photo</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.btnSecondary} onPress={pickPhoto}>
              <Text style={styles.btnSecondaryText}>Choose from library</Text>
            </TouchableOpacity>
            {photo && (
              <TouchableOpacity style={styles.btnPrimary} onPress={park}>
                <Text style={styles.btnPrimaryText}>
                  POST /park — find my spot →
                </Text>
              </TouchableOpacity>
            )}
          </>
        )}

        {step === "matching" && (
          <View style={styles.loadingRow}>
            <ActivityIndicator color="#44403c" />
            <Text style={styles.loadingText}>Running SIFT localization…</Text>
          </View>
        )}

        {step === "parked" && (
          <>
            <View style={styles.sessionBox}>
              <Text style={styles.sessionLabel}>session_id</Text>
              <Text style={styles.sessionId}>{sessionId}</Text>
            </View>
            <TouchableOpacity style={styles.btnPrimary} onPress={find}>
              <Text style={styles.btnPrimaryText}>GET /find/{sessionId} →</Text>
            </TouchableOpacity>
          </>
        )}

        {step === "locating" && (
          <View style={styles.loadingRow}>
            <ActivityIndicator color="#44403c" />
            <Text style={styles.loadingText}>Querying backend…</Text>
          </View>
        )}

        {step === "found" && (
          <>
            <View style={styles.resultBox}>
              <Text style={styles.resultTitle}>
                Your car: <Text style={styles.resultSpot}>{foundSpot}</Text>
              </Text>
              {confidence !== null && (
                <Text style={styles.resultConfidence}>
                  Confidence: {(confidence * 100).toFixed(0)}%
                </Text>
              )}
              {similarityScore !== null && (
                <Text style={styles.resultConfidence}>
                  Similarity score: {similarityScore.toFixed(3)}
                </Text>
              )}
            </View>

            {/* Highlight found spot on the coordinate-accurate lot map */}
            <Text style={styles.mapLabel}>Parking map — amber = your car</Text>
            <View style={styles.mapWrap}>
              <LotMap
                layout={layout}
                highlightSpot={foundSpot}
                dimUnhighlighted={!!foundSpot}
              />
            </View>

            <TouchableOpacity style={styles.btnSecondary} onPress={reset}>
              <Text style={styles.btnSecondaryText}>New session</Text>
            </TouchableOpacity>
          </>
        )}
      </View>
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
  title: { fontSize: 20, fontWeight: "600", color: "#1c1917", marginBottom: 8 },
  subtitle: {
    fontSize: 14,
    color: "#78716c",
    lineHeight: 22,
    marginBottom: 20,
  },
  preview: { width: "100%", height: 200, borderRadius: 12, marginBottom: 16 },
  photoPlaceholder: {
    width: "100%",
    height: 160,
    borderRadius: 12,
    borderWidth: 2,
    borderStyle: "dashed",
    borderColor: "#d4d0cb",
    backgroundColor: "#f5f5f4",
    justifyContent: "center",
    alignItems: "center",
    marginBottom: 16,
  },
  photoPlaceholderText: { fontSize: 14, color: "#a8a29e" },
  error: { fontSize: 13, color: "#c0392b", marginBottom: 12 },
  buttonGroup: { gap: 10 },
  btnPrimary: {
    backgroundColor: "#ffffff",
    borderWidth: 1,
    borderColor: "#a8a29e",
    borderRadius: 10,
    paddingVertical: 14,
    alignItems: "center",
    marginBottom: 6,
  },
  btnPrimaryText: { fontSize: 14, fontWeight: "500", color: "#1c1917" },
  btnSecondary: {
    borderWidth: 1,
    borderColor: "#e7e5e4",
    borderRadius: 10,
    paddingVertical: 12,
    alignItems: "center",
    marginBottom: 6,
  },
  btnSecondaryText: { fontSize: 14, color: "#78716c" },
  loadingRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 10,
    paddingVertical: 16,
  },
  loadingText: { fontSize: 14, color: "#78716c" },
  sessionBox: {
    backgroundColor: "#f5f5f4",
    borderRadius: 10,
    padding: 14,
    marginBottom: 6,
  },
  sessionLabel: { fontSize: 11, color: "#a8a29e", marginBottom: 4 },
  sessionId: { fontSize: 13, fontWeight: "600", color: "#1c1917" },
  resultBox: {
    backgroundColor: "#fffbeb",
    borderWidth: 1,
    borderColor: "#e89600",
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
  },
  resultTitle: {
    fontSize: 16,
    fontWeight: "500",
    color: "#7d5200",
    marginBottom: 4,
  },
  resultSpot: { fontWeight: "700" },
  resultConfidence: { fontSize: 13, color: "#7d5200" },
  mapLabel: { fontSize: 13, color: "#78716c", marginBottom: 10 },
  mapWrap: { marginBottom: 16 },
});
