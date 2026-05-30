import { useState } from "react";
import { NavigationContainer } from "@react-navigation/native";
import { createBottomTabNavigator } from "@react-navigation/bottom-tabs";
import { SafeAreaProvider } from "react-native-safe-area-context";
import { StyleSheet, View } from "react-native";
import OwnerSetupScreen from "./screens/OwnerSetupScreen";
import OccupancyMapScreen from "./screens/OccupancyMapScreen";
import FindMyCarScreen from "./screens/FindMyCarScreen";

const Tab = createBottomTabNavigator();

function SetupIcon({ color }) {
  return (
    <View style={styles.iconFrame}>
      <View style={[styles.gearToothVertical, { backgroundColor: color }]} />
      <View style={[styles.gearToothHorizontal, { backgroundColor: color }]} />
      <View style={[styles.gearRing, { borderColor: color }]}>
        <View style={[styles.gearDot, { backgroundColor: color }]} />
      </View>
    </View>
  );
}

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

  return (
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
            name="Setup"
            options={{
              title: "Owner Setup",
              tabBarLabel: "Setup",
              tabBarIcon: ({ color }) => <SetupIcon color={color} />,
            }}
          >
            {() => <OwnerSetupScreen onLayoutReady={setLayout} />}
          </Tab.Screen>

          <Tab.Screen
            name="Map"
            options={{
              title: "Live Occupancy",
              tabBarLabel: "Live Map",
              tabBarIcon: ({ color }) => <MapIcon color={color} />,
            }}
          >
            {() => <OccupancyMapScreen layout={layout} />}
          </Tab.Screen>

          <Tab.Screen
            name="Find"
            options={{
              title: "Find My Car",
              tabBarLabel: "Find Car",
              tabBarIcon: ({ color }) => <FindIcon color={color} />,
            }}
          >
            {() => <FindMyCarScreen layout={layout} />}
          </Tab.Screen>
        </Tab.Navigator>
      </NavigationContainer>
    </SafeAreaProvider>
  );
}

const styles = StyleSheet.create({
  iconFrame: {
    width: 24,
    height: 24,
    alignItems: "center",
    justifyContent: "center",
  },
  gearRing: {
    width: 16,
    height: 16,
    borderWidth: 2,
    borderRadius: 8,
    alignItems: "center",
    justifyContent: "center",
  },
  gearDot: {
    width: 5,
    height: 5,
    borderRadius: 3,
  },
  gearToothVertical: {
    position: "absolute",
    width: 4,
    height: 22,
    borderRadius: 2,
  },
  gearToothHorizontal: {
    position: "absolute",
    width: 22,
    height: 4,
    borderRadius: 2,
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
