import { useState } from "react";
import { Dimensions, StyleSheet, View } from "react-native";
import { Gesture, GestureDetector } from "react-native-gesture-handler";
import Animated, {
  runOnJS,
  useAnimatedReaction,
  useAnimatedStyle,
  useSharedValue,
  withTiming,
} from "react-native-reanimated";
import Svg, {
  Image as SvgImage,
  Polygon,
  Rect,
  Text as SvgText,
} from "react-native-svg";

// Mirrors the web LeafletMap palette.
const SPOT_COLORS = {
  free: "#1a7a4a",
  occupied: "#c0392b",
  unknown: "#e7e5e4",
  highlight: "#e89600",
};

const MIN_SCALE = 1;
const MAX_SCALE = 5;
// Above this zoom level one-finger drags pan the map instead of
// scrolling the surrounding screen.
const PAN_THRESHOLD = 1.05;

function clamp(value, lo, hi) {
  "worklet";
  return Math.min(Math.max(value, lo), hi);
}

function centroid(corners) {
  const [sx, sy] = corners.reduce(
    ([ax, ay], [x, y]) => [ax + x, ay + y],
    [0, 0],
  );
  return [sx / corners.length, sy / corners.length];
}

/**
 * Coordinate-accurate parking lot map.
 *
 * Renders spot quadrilaterals at their exact layout coordinates inside an
 * SVG viewBox, fit-to-screen by default. Pinch zooms (1x–5x); once zoomed,
 * one-finger drag pans like a map app and snaps back inside the viewport.
 */
export default function LotMap({
  layout,
  status,
  highlightSpot,
  dimUnhighlighted,
  onSpotPress,
  selectedSpot,
}) {
  const [containerWidth, setContainerWidth] = useState(0);
  const [zoomed, setZoomed] = useState(false);

  const scale = useSharedValue(1);
  const savedScale = useSharedValue(1);
  const tx = useSharedValue(0);
  const ty = useSharedValue(0);
  const savedTx = useSharedValue(0);
  const savedTy = useSharedValue(0);

  useAnimatedReaction(
    () => scale.value > PAN_THRESHOLD,
    (isZoomed, prev) => {
      if (isZoomed !== prev) runOnJS(setZoomed)(isZoomed);
    },
  );

  const canvasW = layout?.canvas?.width ?? 600;
  const canvasH = layout?.canvas?.height ?? 400;

  // Fit the full lot on screen at base zoom: bounded by container width and
  // ~55% of the window height.
  const maxHeight = Dimensions.get("window").height * 0.55;
  const fitScale = containerWidth
    ? Math.min(containerWidth / canvasW, maxHeight / canvasH)
    : 0;
  const displayW = canvasW * fitScale;
  const displayH = canvasH * fitScale;

  // Stroke/font sizes are specified in canvas units — compensate so they
  // render at a constant on-screen size regardless of canvas resolution.
  const unit = fitScale > 0 ? 1 / fitScale : 1;

  const pinch = Gesture.Pinch()
    .onUpdate((e) => {
      scale.value = clamp(savedScale.value * e.scale, MIN_SCALE, MAX_SCALE);
    })
    .onEnd(() => {
      savedScale.value = scale.value;
      const maxX = (displayW * (scale.value - 1)) / 2;
      const maxY = (displayH * (scale.value - 1)) / 2;
      const cx = clamp(tx.value, -maxX, maxX);
      const cy = clamp(ty.value, -maxY, maxY);
      tx.value = withTiming(cx);
      ty.value = withTiming(cy);
      savedTx.value = cx;
      savedTy.value = cy;
    });

  const pan = Gesture.Pan()
    .enabled(zoomed)
    .onUpdate((e) => {
      const maxX = (displayW * (scale.value - 1)) / 2;
      const maxY = (displayH * (scale.value - 1)) / 2;
      tx.value = clamp(savedTx.value + e.translationX, -maxX, maxX);
      ty.value = clamp(savedTy.value + e.translationY, -maxY, maxY);
    })
    .onEnd(() => {
      savedTx.value = tx.value;
      savedTy.value = ty.value;
    });

  const gesture = Gesture.Simultaneous(pinch, pan);

  const animatedStyle = useAnimatedStyle(() => ({
    transform: [
      { translateX: tx.value },
      { translateY: ty.value },
      { scale: scale.value },
    ],
  }));

  if (!layout?.spots?.length) return null;

  return (
    <View
      style={styles.frame}
      onLayout={(e) => setContainerWidth(e.nativeEvent.layout.width)}
    >
      {fitScale > 0 && (
        <View style={[styles.viewport, { width: displayW, height: displayH }]}>
          <GestureDetector gesture={gesture}>
            <Animated.View
              style={[{ width: displayW, height: displayH }, animatedStyle]}
            >
              <Svg
                width={displayW}
                height={displayH}
                viewBox={`0 0 ${canvasW} ${canvasH}`}
              >
                {layout.background_image ? (
                  <SvgImage
                    href={{ uri: layout.background_image }}
                    x={0}
                    y={0}
                    width={canvasW}
                    height={canvasH}
                    preserveAspectRatio="xMidYMid slice"
                  />
                ) : (
                  <Rect
                    x={0}
                    y={0}
                    width={canvasW}
                    height={canvasH}
                    fill="#f5f5f4"
                  />
                )}

                {layout.spots.map((spot) => {
                  const occ = status?.[spot.spot_id] ?? "unknown";
                  const isHighlight = highlightSpot === spot.spot_id;
                  const isDimmed =
                    dimUnhighlighted && highlightSpot && !isHighlight;
                  const isSelected = selectedSpot === spot.spot_id;
                  const fill = isHighlight
                    ? SPOT_COLORS.highlight
                    : SPOT_COLORS[occ];
                  return (
                    <Polygon
                      key={spot.spot_id}
                      points={spot.corners
                        .map(([x, y]) => `${x},${y}`)
                        .join(" ")}
                      fill={fill}
                      fillOpacity={isDimmed ? 0.25 : 0.85}
                      stroke={isSelected ? "#1c1917" : "#ffffff"}
                      strokeWidth={(isSelected ? 3 : 1.5) * unit}
                      onPress={
                        onSpotPress
                          ? () => onSpotPress(spot.spot_id)
                          : undefined
                      }
                    />
                  );
                })}

                {layout.spots.map((spot) => {
                  const [cx, cy] = centroid(spot.corners);
                  const occ = status?.[spot.spot_id] ?? "unknown";
                  const isHighlight = highlightSpot === spot.spot_id;
                  const labelColor =
                    occ === "unknown" && !isHighlight ? "#78716c" : "#ffffff";
                  return (
                    <SvgText
                      key={`${spot.spot_id}-label`}
                      x={cx}
                      y={cy}
                      fill={labelColor}
                      fontSize={11 * unit}
                      fontWeight="600"
                      textAnchor="middle"
                      alignmentBaseline="middle"
                    >
                      {spot.spot_id.replace("spot_", "P")}
                    </SvgText>
                  );
                })}
              </Svg>
            </Animated.View>
          </GestureDetector>
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  frame: {
    width: "100%",
    alignItems: "center",
  },
  viewport: {
    overflow: "hidden",
    borderRadius: 12,
    borderWidth: 1,
    borderColor: "#e7e5e4",
  },
});
