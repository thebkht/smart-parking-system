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

export default function OwnerSetupScreen({ onLayoutReady }) {
  const [images, setImages] = useState([]);
  const [step, setStep] = useState("idle");
  const [layout, setLayout] = useState(null);

  const pickImages = async () => {
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsMultipleSelection: true,
      quality: 0.8,
    });
    if (!result.canceled) {
      setImages(result.assets);
      setStep("ready");
    }
  };

  const takePhoto = async () => {
    const perm = await ImagePicker.requestCameraPermissionsAsync();
    if (!perm.granted) {
      Alert.alert("Permission required", "Camera access is needed.");
      return;
    }
    const result = await ImagePicker.launchCameraAsync({ quality: 0.8 });
    if (!result.canceled) {
      setImages((prev) => [...prev, result.assets[0]]);
      setStep("ready");
    }
  };

  const submit = async () => {
    if (images.length < 4) {
      Alert.alert("Too few photos", "Upload at least 4 photos for reliable SfM.");
      return;
    }
    setStep("processing");
    const form = new FormData();
    images.forEach((img, i) => {
      form.append("images", {
        uri: img.uri,
        name: `photo_${i}.jpg`,
        type: "image/jpeg",
      });
    });
    try {
      const { data } = await api.post("/layout", form, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setLayout(data);
      onLayoutReady(data);
    } catch {
      try {
        const { data } = await api.get("/map");
        setLayout(data);
        onLayoutReady(data);
      } catch {
        Alert.alert("Backend unreachable", "Could not load layout from server.");
      }
    }
    setStep("done");
  };

  const reset = () => {
    setImages([]);
    setStep("idle");
    setLayout(null);
  };

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      <Text style={styles.title}>Owner Setup</Text>
      <Text style={styles.subtitle}>
        Upload 4–5 overlapping photos of your parking lot. The SfM pipeline will
        compute a bird's-eye-view layout automatically.
      </Text>

      {/* Image thumbnails */}
      {images.length > 0 && (
        <ScrollView horizontal style={styles.thumbRow}>
          {images.map((img, i) => (
            <Image key={i} source={{ uri: img.uri }} style={styles.thumb} />
          ))}
        </ScrollView>
      )}

      {/* Buttons */}
      {(step === "idle" || step === "ready") && (
        <View style={styles.buttonGroup}>
          <TouchableOpacity style={styles.btnPrimary} onPress={pickImages}>
            <Text style={styles.btnPrimaryText}>Select photos from library</Text>
          </TouchableOpacity>
          <TouchableOpacity style={styles.btnSecondary} onPress={takePhoto}>
            <Text style={styles.btnSecondaryText}>Take a photo</Text>
          </TouchableOpacity>
          {images.length >= 4 && (
            <TouchableOpacity style={styles.btnPrimary} onPress={submit}>
              <Text style={styles.btnPrimaryText}>
                Run SfM layout ({images.length} photos)
              </Text>
            </TouchableOpacity>
          )}
        </View>
      )}

      {step === "processing" && (
        <View style={styles.loadingRow}>
          <ActivityIndicator color="#44403c" />
          <Text style={styles.loadingText}>Running SfM pipeline…</Text>
        </View>
      )}

      {layout && (
        <View style={styles.resultBox}>
          <Text style={styles.resultTitle}>
            Layout ready — {layout.spots?.length ?? 0} spots detected
          </Text>
          <Text style={styles.resultSub}>Source: {layout.spot_source}</Text>
          <TouchableOpacity style={styles.btnSecondary} onPress={reset}>
            <Text style={styles.btnSecondaryText}>Re-upload</Text>
          </TouchableOpacity>
        </View>
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#fafaf9" },
  content: { padding: 20 },
  title: { fontSize: 20, fontWeight: "600", color: "#1c1917", marginBottom: 8 },
  subtitle: { fontSize: 14, color: "#78716c", lineHeight: 22, marginBottom: 20 },
  thumbRow: { marginBottom: 16 },
  thumb: { width: 80, height: 80, borderRadius: 8, marginRight: 8 },
  buttonGroup: { gap: 10 },
  btnPrimary: {
    backgroundColor: "#ffffff",
    borderWidth: 1,
    borderColor: "#a8a29e",
    borderRadius: 10,
    paddingVertical: 14,
    alignItems: "center",
    marginBottom: 10,
  },
  btnPrimaryText: { fontSize: 14, fontWeight: "500", color: "#1c1917" },
  btnSecondary: {
    borderWidth: 1,
    borderColor: "#e7e5e4",
    borderRadius: 10,
    paddingVertical: 12,
    alignItems: "center",
    marginBottom: 10,
  },
  btnSecondaryText: { fontSize: 14, color: "#78716c" },
  loadingRow: { flexDirection: "row", alignItems: "center", gap: 10, paddingVertical: 16 },
  loadingText: { fontSize: 14, color: "#78716c" },
  resultBox: {
    marginTop: 24,
    padding: 16,
    backgroundColor: "#f5f5f4",
    borderRadius: 12,
  },
  resultTitle: { fontSize: 15, fontWeight: "600", color: "#1c1917", marginBottom: 4 },
  resultSub: { fontSize: 13, color: "#78716c", marginBottom: 12 },
});
