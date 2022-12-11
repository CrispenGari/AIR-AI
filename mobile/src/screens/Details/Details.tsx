import {
  View,
  Text,
  StatusBar,
  Image,
  TouchableOpacity,
  SafeAreaView,
  ScrollView,
  Dimensions,
  TextInput,
  KeyboardAvoidingView,
  TouchableWithoutFeedback,
  Keyboard,
  Alert,
} from "react-native";
import * as Clipboard from "expo-clipboard";
import React from "react";
import { COLORS, FONTS } from "../../constants";
import { Entypo, AntDesign } from "@expo/vector-icons";

const copyToClipboard = async (text) => {
  await Clipboard.setStringAsync(text);
  Alert.alert(
    "CR",
    `Copied text "${text}" to clipboard.`,
    [
      {
        style: "destructive",
        text: "OK",
        onPress: () => {},
      },
    ],
    {
      cancelable: false,
    }
  );
};

const Details = ({
  navigation,
  route: {
    params: { data, image },
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
          <Entypo name="chevron-small-left" size={30} color={COLORS.green} />
          <Text
            style={{
              fontFamily: FONTS.NunitoSansRegular,
              fontSize: 20,
              marginLeft: 0,
              color: COLORS.green,
            }}
          >
            Home
          </Text>
        </TouchableOpacity>
      ),
      headerTitle: "Detected Text",
    });
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
        >
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
          <KeyboardAvoidingView
            style={{
              width: "100%",
              padding: 10,
            }}
          >
            <Text
              style={{
                padding: 5,
                fontFamily: FONTS.NunitoSansRegular,
                fontSize: 30,
                textAlign: "center",
              }}
            >
              Detected Text
            </Text>
            {data?.results?.map((result, index) => (
              <View
                key={index}
                style={{
                  backgroundColor: "white",
                  width: "100%",
                  paddingHorizontal: 10,
                  paddingVertical: 5,
                  flexDirection: "row",
                  alignItems: "center",
                }}
              >
                <TextInput
                  style={{
                    backgroundColor: "#F5F5F5",
                    padding: 5,
                    fontFamily: FONTS.NunitoSansRegular,
                    flex: 1,
                  }}
                  multiline
                  value={result}
                  editable={false}
                />
                <TouchableOpacity
                  style={{
                    marginLeft: 10,
                  }}
                  activeOpacity={0.5}
                  onPress={() => copyToClipboard(result)}
                >
                  <AntDesign name="copy1" size={24} color={COLORS.green} />
                </TouchableOpacity>
              </View>
            ))}
          </KeyboardAvoidingView>
          <TouchableOpacity
            onPress={() => {
              navigation.goBack();
            }}
            style={{
              marginVertical: 30,
              backgroundColor: COLORS.green,
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
          <View style={{ height: 200 }} />
          <Text
            style={{
              margin: 20,
              color: COLORS.green,
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
