import { View, Text, Dimensions } from "react-native";
import React, { useEffect } from "react";
import { BarChart } from "react-native-chart-kit";
import { COLORS, FONTS } from "../../constants";
import { Predicted } from "../../graphql/generated/graphql";
import { ScrollView } from "react-native-gesture-handler";
interface Props {
  predictions: Predicted[];
}
const Bar: React.FunctionComponent<Props> = ({ predictions }) => {
  const [labels, setLabels] = React.useState([]);
  const [data, setData] = React.useState([]);

  useEffect(() => {
    let mounted = true;
    if (mounted) {
      setData(predictions?.map((pred) => (pred.probability * 100).toFixed(0)));
      setLabels(predictions?.map((pred) => pred.className.toLowerCase()));
    }
    return () => {
      mounted = false;
    };
  }, []);
  return (
    <ScrollView style={{ marginTop: 20 }} horizontal>
      <View>
        <Text
          style={{
            color: COLORS.main,
            textAlign: "center",
            marginBottom: 10,
            fontFamily: FONTS.NunitoSansRegular,
            fontSize: 20,
            letterSpacing: 2,
          }}
        >
          Prediction Distribution
        </Text>
        <BarChart
          style={{ marginTop: 10, width: "100%", justifyContent: "center" }}
          data={{
            labels,
            datasets: [
              {
                data,
                withDots: true,
                color: (opacity = 1) => COLORS.main,
                withScrollableDot: true,
              },
            ],
          }}
          fromZero
          width={Dimensions.get("screen").width}
          height={Dimensions.get("screen").height * 0.4}
          yAxisLabel="  "
          yAxisSuffix=""
          showBarTops={false}
          chartConfig={{
            backgroundGradientFrom: "white",
            backgroundGradientFromOpacity: 1,
            backgroundGradientTo: "white",
            backgroundGradientToOpacity: 0.5,
            paddingRight: 0,
            horizontalOffset: 0,
            color: (opacity = 1) => `rgba(57, 91, 100, ${opacity})`,
            strokeWidth: 1,
            barPercentage: 0.4,
            propsForDots: {
              r: "2",
              strokeWidth: "2",
              stroke: "white",
            },
            fillShadowGradient: COLORS.main,
            fillShadowGradientOpacity: 1,
            fillShadowGradientTo: COLORS.main,
            scrollableDotFill: "%",
            formatTopBarValue: (val) => `${val}%`,
            formatYLabel: (val) => `${Number(val).toFixed(0)}%`,
          }}
          verticalLabelRotation={60}
          horizontalLabelRotation={-45}
          showValuesOnTopOfBars
          yLabelsOffset={10}
        />
      </View>
    </ScrollView>
  );
};

export default Bar;
