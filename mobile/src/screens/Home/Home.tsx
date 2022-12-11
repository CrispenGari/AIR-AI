import {
  View,
  Text,
  StatusBar,
  Image,
  TouchableOpacity,
  SafeAreaView,
  Alert,
  ImageBackground,
} from "react-native";
import React, { useState } from "react";
import { COLORS, FONTS } from "../../constants";
import { Ionicons, AntDesign } from "@expo/vector-icons";
import { useMediaPermissions } from "../../hooks";
import * as ImagePicker from "expo-image-picker";
import * as Camera from "expo-camera";
import { useClassifyAnimalMutation } from "../../graphql/generated/graphql";
import Requesting from "../../components/Requesting/Requesting";
import { generateRNFile } from "../../utils";
import { AppNavProps } from "../../params";
const Home: React.FunctionComponent<AppNavProps<"Home">> = ({ navigation }) => {
  const { camera, library } = useMediaPermissions();
  const [image, setImage] = useState(null);
  const [predict, { loading, data }] = useClassifyAnimalMutation({
    fetchPolicy: "network-only",
  });
  const openCamera = async () => {
    if (!camera) {
      Alert.alert(
        "AIR",
        "AIR tool does not have permission to access your camera.",
        [
          {
            style: "default",
            text: "Allow Permission",
            onPress: async () => {
              await Camera.requestCameraPermissionsAsync();
              return;
            },
          },
          {
            style: "destructive",
            text: "CANCEL",
            onPress: () => {},
          },
        ]
      );
      return;
    }
    const { assets, canceled } = await ImagePicker.launchCameraAsync({
      allowsEditing: true,
      aspect: [1, 1],
      base64: false,
      quality: 1,
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsMultipleSelection: false,
    });

    if (!canceled) {
      setImage({
        uri: assets[0].uri,
        name: assets[0].fileName,
      });
    }
  };
  const selectImage = async () => {
    if (!library) {
      Alert.alert(
        "AIR",
        "AIR tool does not have permission to access your photos.",
        [
          {
            style: "default",
            text: "Allow Access to all Photos",
            onPress: async () => {
              await ImagePicker.requestMediaLibraryPermissionsAsync();
              return;
            },
          },
          {
            style: "destructive",
            text: "CANCEL",
            onPress: () => {},
          },
        ]
      );
      return;
    }

    const { assets, canceled } = await ImagePicker.launchImageLibraryAsync({
      allowsEditing: true,
      base64: false,
      quality: 1,
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsMultipleSelection: false,
    });

    if (!canceled) {
      setImage({
        uri: assets[0].uri,
        name: assets[0].fileName,
      });
    }
  };

  const recognizeImage = async () => {
    if (!!!image) {
      Alert.alert("AIR", "please select an image with text first.", [
        {
          style: "destructive",
          text: "CANCEL",
          onPress: () => {},
        },
      ]);
      return;
    }

    const _image = generateRNFile(image);
    await predict({
      variables: {
        input: {
          image: _image,
        },
      },
    });
  };

  React.useEffect(() => {
    let mounted = true;
    if (mounted && !!data?.predictAnimal?.prediction) {
      navigation.navigate("Details", {
        image,
        prediction: data.predictAnimal.prediction,
      });
    }
    return () => {
      mounted = false;
    };
  }, [data]);
  return (
    <View
      style={{
        flex: 1,
        backgroundColor: COLORS.main,
      }}
    >
      {loading ? <Requesting /> : null}
      <StatusBar barStyle={"light-content"} />
      <SafeAreaView style={{ flex: 1 }}>
        <View
          style={{
            marginBottom: 30,
            flex: 0.8,
          }}
        >
          {!!!image ? (
            <TouchableOpacity
              activeOpacity={0.7}
              onPress={selectImage}
              style={{
                width: "100%",
                height: "70%",
                justifyContent: "center",
                alignItems: "center",
                backgroundColor: COLORS.naive,
              }}
            >
              <ImageBackground
                style={{
                  flex: 1,
                  justifyContent: "center",
                  alignItems: "center",
                  padding: 10,
                  width: "100%",
                }}
                source={{
                  uri: Image.resolveAssetSource(
                    require("../../../assets/cover.jpg")
                  ).uri,
                }}
              >
                <Text
                  style={{
                    padding: 10,
                    textAlign: "center",
                    color: COLORS.main,
                    fontFamily: FONTS.NunitoSansBlack,
                    fontSize: 20,
                    letterSpacing: 2,
                  }}
                >
                  Select Image Or Take a photo.
                </Text>
              </ImageBackground>
            </TouchableOpacity>
          ) : (
            <Image
              source={{
                uri: image?.uri,
              }}
              style={{
                width: "100%",
                height: "70%",
                resizeMode: "contain",
              }}
            />
          )}
          <Text
            style={{
              padding: 10,
              textAlign: "center",
              color: "white",
              fontFamily: FONTS.NunitoSansRegular,
            }}
          >
            Choose an image of an animal or take a new one.
          </Text>
          <View
            style={{
              margin: 10,
              padding: 20,
              backgroundColor: COLORS.naive,
              borderRadius: 10,
              flexDirection: "row",
              alignItems: "center",
              justifyContent: "space-evenly",
            }}
          >
            <TouchableOpacity
              style={{
                backgroundColor: COLORS.orange,
                padding: 10,
                width: 50,
                height: 50,
                justifyContent: "center",
                alignItems: "center",
                borderRadius: 50,
              }}
              activeOpacity={0.7}
              onPress={openCamera}
            >
              <Ionicons name="camera" size={24} color="white" />
            </TouchableOpacity>
            <TouchableOpacity
              style={{
                backgroundColor: COLORS.orange,
                padding: 10,
                width: 50,
                height: 50,
                justifyContent: "center",
                alignItems: "center",
                borderRadius: 50,
              }}
              activeOpacity={0.7}
              onPress={selectImage}
            >
              <AntDesign name="picture" size={24} color="white" />
            </TouchableOpacity>
          </View>
        </View>

        <View
          style={{
            justifyContent: "space-between",
            flex: 0.2,
            width: "100%",
            alignItems: "center",
          }}
        >
          <TouchableOpacity
            onPress={recognizeImage}
            style={{
              marginVertical: 30,
              backgroundColor: COLORS.orange,
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
              RECOGNIZE
            </Text>
          </TouchableOpacity>
          <Text
            style={{
              margin: 20,
              color: "white",
              textAlign: "center",
              fontFamily: FONTS.NunitoSansItalic,
            }}
          >
            AI tool developed by @crispengari.
          </Text>
        </View>
      </SafeAreaView>
    </View>
  );
};

export default Home;
