import "react-native-gesture-handler";
import { StyleSheet, Text, View, StatusBar } from "react-native";
import * as ImagePicker from "expo-image-picker";
import Routes from "./src/routes";
import { useFonts } from "expo-font";
import { Loading } from "./src/components";
import { Fonts } from "./src/constants";
import GraphQLProvider from './src/providers/GraphQLProvider'

const App = () => {
  const [loaded] = useFonts(Fonts);

  if (!loaded) {
    return <Loading />;
  }
  return <GraphQLProvider><Routes /></GraphQLProvider>;
};

export default App;
