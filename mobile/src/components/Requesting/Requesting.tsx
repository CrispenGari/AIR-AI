import { View, Text } from "react-native";
import React from "react";
import { COLORS, FONTS } from "../../constants";

const Requesting = () => {
  return (
    <View
      style={{
        backgroundColor: "rgba(0, 0, 0, .5)",
        zIndex: 10,
        position: "absolute",
        top: 0,
        bottom: 0,
        right: 0,
        left: 0,
        justifyContent: "center",
        alignItems: "center",
        padding: 10,
      }}
    >
      <Text
        style={{
          color: "white",
          fontFamily: FONTS.NunitoSansRegular,
          fontSize: 20,
          letterSpacing: 2,
        }}
      >
        detecting...
      </Text>
    </View>
  );
};

export default Requesting;
