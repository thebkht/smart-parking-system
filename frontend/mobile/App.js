import { useState, useEffect, useCallback } from "react";
import { NavigationContainer } from "@react-navigation/native";
import { createBottomTabNavigator } from "@react-navigation/bottom-tabs";
import { SafeAreaProvider } from "react-native-safe-area-context";
import { GestureHandlerRootView } from "react-native-gesture-handler";
import { StyleSheet, View } from "react-native";
import OccupancyMapScreen from "./screens/OccupancyMapScreen";
import FindMyCarScreen from "./screens/FindMyCarScreen";
import { api, API_BASE } from "./api";
import { normalizeLayout } from "./layout";

const Tab = createBottomTabNavigator();

function MapIcon({ color }) {
  return (
    <View style={styles.iconFrame}>
      <View style={[styles.mapLine, { borderColor: color }]} />
      <View style={[styles.mapPin, { borderColor: color }]}>
        <View style={[styles.mapPinDot, { backgroundColor: color }]} />
      </View>
    </View>
  );
}

function FindIcon({ color }) {
  return (
    <View style={styles.iconFrame}>
      <View style={[styles.searchRing, { borderColor: color }]} />
      <View style={[styles.searchHandle, { backgroundColor: color }]} />
    </View>
  );
}

export default function App() {
  const [layout, setLayout] = useState(null);
  const [layoutError, setLayoutError] = useState(null);

  // Owner setup lives on the web app — the mobile app consumes the
  // published layout from the backend.
  const loadLayout = useCallback(async () => {
    setLayoutError(null);
    try {
      const { data } = await api.get("/map");
      setLayout(normalizeLayout(data, { apiBase: API_BASE }));
    } catch (error) {
      setLayoutError(
        error?.response?.status === 404
          ? "No layout published yet."
          : "Backend unreachable.",
      );
    }
  }, []);

  useEffect(() => {
    loadLayout();
  }, [loadLayout]);

  return (
    <GestureHandlerRootView style={styles.root}>
      <SafeAreaProvider>
        <NavigationContainer>
          <Tab.Navigator
            screenOptions={{
              headerStyle: { backgroundColor: "#fafaf9" },
              headerTintColor: "#1c1917",
              headerTitleStyle: { fontWeight: "600" },
              tabBarStyle: {
                backgroundColor: "#fafaf9",
                borderTopColor: "#e7e5e4",
              },
              tabBarActiveTintColor: "#1c1917",
              tabBarInactiveTintColor: "#a8a29e",
            }}
          >
            <Tab.Screen
              name="Map"
              options={{
                title: "Live Occupancy",
                tabBarLabel: "Live Map",
                tabBarIcon: ({ color }) => <MapIcon color={color} />,
              }}
            >
              {() => (
                <OccupancyMapScreen
                  layout={layout}
                  layoutError={layoutError}
                  onRetry={loadLayout}
                />
              )}
            </Tab.Screen>

            <Tab.Screen
              name="Find"
              options={{
                title: "Find My Car",
                tabBarLabel: "Find Car",
                tabBarIcon: ({ color }) => <FindIcon color={color} />,
              }}
            >
              {() => (
                <FindMyCarScreen
                  layout={layout}
                  layoutError={layoutError}
                  onRetry={loadLayout}
                />
              )}
            </Tab.Screen>
          </Tab.Navigator>
        </NavigationContainer>
      </SafeAreaProvider>
    </GestureHandlerRootView>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1 },
  iconFrame: {
    width: 24,
    height: 24,
    alignItems: "center",
    justifyContent: "center",
  },
  mapLine: {
    position: "absolute",
    width: 18,
    height: 14,
    borderWidth: 2,
    borderRadius: 3,
    transform: [{ rotate: "-8deg" }],
  },
  mapPin: {
    width: 10,
    height: 10,
    borderWidth: 2,
    borderRadius: 5,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "#fafaf9",
  },
  mapPinDot: {
    width: 3,
    height: 3,
    borderRadius: 2,
  },
  searchRing: {
    width: 15,
    height: 15,
    borderWidth: 2,
    borderRadius: 8,
    transform: [{ translateX: -2 }, { translateY: -2 }],
  },
  searchHandle: {
    position: "absolute",
    width: 10,
    height: 2,
    borderRadius: 1,
    transform: [{ translateX: 6 }, { translateY: 6 }, { rotate: "45deg" }],
  },
});
