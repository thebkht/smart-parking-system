import { useState } from "react";
import { NavigationContainer } from "@react-navigation/native";
import { createBottomTabNavigator } from "@react-navigation/bottom-tabs";
import { SafeAreaProvider } from "react-native-safe-area-context";
import { Text } from "react-native";
import OwnerSetupScreen from "./screens/OwnerSetupScreen";
import OccupancyMapScreen from "./screens/OccupancyMapScreen";
import FindMyCarScreen from "./screens/FindMyCarScreen";

const Tab = createBottomTabNavigator();

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
              tabBarIcon: ({ color }) => (
                <Text style={{ color, fontSize: 18 }}>⚙️</Text>
              ),
            }}
          >
            {() => <OwnerSetupScreen onLayoutReady={setLayout} />}
          </Tab.Screen>

          <Tab.Screen
            name="Map"
            options={{
              title: "Live Occupancy",
              tabBarLabel: "Live Map",
              tabBarIcon: ({ color }) => (
                <Text style={{ color, fontSize: 18 }}>🅿️</Text>
              ),
            }}
          >
            {() => <OccupancyMapScreen layout={layout} />}
          </Tab.Screen>

          <Tab.Screen
            name="Find"
            options={{
              title: "Find My Car",
              tabBarLabel: "Find Car",
              tabBarIcon: ({ color }) => (
                <Text style={{ color, fontSize: 18 }}>🔍</Text>
              ),
            }}
          >
            {() => <FindMyCarScreen layout={layout} />}
          </Tab.Screen>
        </Tab.Navigator>
      </NavigationContainer>
    </SafeAreaProvider>
  );
}
