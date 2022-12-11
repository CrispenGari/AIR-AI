import {
  View,
  Text,
  StatusBar,
  Image,
  TouchableOpacity,
  SafeAreaView,
  ScrollView,
  Dimensions,
  TouchableWithoutFeedback,
  Keyboard,
} from "react-native";
import React from "react";
import { COLORS, FONTS } from "../../constants";
import { Entypo, AntDesign } from "@expo/vector-icons";
import { AppNavProps } from "../../params";
import { BarChart, Table } from "../../components";

const Details: React.FunctionComponent<AppNavProps<"Details">> = ({
  navigation,
  route: {
    params: { prediction, image },
  },
}) => {
  React.useLayoutEffect(() => {
    navigation.setOptions({
      headerShown: true,
      headerTitleStyle: {
        fontFamily: FONTS.NunitoSansRegular,
        fontSize: 20,
      },
      headerLeft: () => (
        <TouchableOpacity
          style={{
            marginHorizontal: 10,
            flexDirection: "row",
            alignItems: "center",
          }}
          activeOpacity={0.7}
          onPress={() => navigation.goBack()}
        >
          <Entypo name="chevron-small-left" size={30} color={COLORS.orange} />
          <Text
            style={{
              fontFamily: FONTS.NunitoSansRegular,
              fontSize: 20,
              marginLeft: 0,
              color: COLORS.orange,
            }}
          >
            Home
          </Text>
        </TouchableOpacity>
      ),
      headerTitle: prediction.topPrediction.className,
    });
  }, []);

  const topPredictionsHeader = ["Class Name", "Label", "Probability"];
  const [topPredictions, setTopPredictions] = React.useState([[]]);

  React.useEffect(() => {
    let mounted = true;

    if (mounted) {
      setTopPredictions([
        [
          prediction.topPrediction?.className,
          prediction?.topPrediction?.label,
          prediction?.topPrediction?.probability?.toFixed(2),
        ],
      ]);
    }
    return () => {
      mounted = false;
    };
  }, []);

  return (
    <SafeAreaView style={{ flex: 1 }}>
      <StatusBar barStyle={"dark-content"} />
      <TouchableWithoutFeedback style={{ flex: 1 }} onPress={Keyboard.dismiss}>
        <ScrollView
          style={{ flex: 1 }}
          contentContainerStyle={{
            alignItems: "center",
          }}
          showsVerticalScrollIndicator={false}
        >
          <Table
            title={`Top Prediction says this image is a ${
              prediction.topPrediction.className
            } with ${(prediction.topPrediction.probability * 100).toFixed(
              0
            )}% confidence.`}
            tableData={topPredictions}
            tableHead={topPredictionsHeader}
          />
          <Image
            source={{
              uri: image.uri,
            }}
            style={{
              width: "100%",
              height: Dimensions.get("screen").height * 0.4,
              resizeMode: "contain",
            }}
          />
          <BarChart predictions={prediction.predictions} />
          <TouchableOpacity
            onPress={() => {
              navigation.goBack();
            }}
            style={{
              marginVertical: 30,
              backgroundColor: COLORS.naive,
              width: "90%",
              paddingHorizontal: 20,
              paddingVertical: 15,
              justifyContent: "center",
              alignItems: "center",
              borderRadius: 999,
            }}
            activeOpacity={0.7}
          >
            <Text
              style={{
                color: "white",
                fontFamily: FONTS.NunitoSansRegular,
                letterSpacing: 2,
                fontSize: 20,
              }}
            >
              NEW IMAGE
            </Text>
          </TouchableOpacity>

          <Text
            style={{
              margin: 20,
              color: COLORS.main,
              textAlign: "center",
              fontFamily: FONTS.NunitoSansItalic,
            }}
          >
            AI tool developed by @crispengari.
          </Text>
        </ScrollView>
      </TouchableWithoutFeedback>
    </SafeAreaView>
  );
};

export default Details;
