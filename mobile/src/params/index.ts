import { StackNavigationProp } from "@react-navigation/stack";
import { RouteProp } from "@react-navigation/native";
import { Prediction } from "../graphql/generated/graphql";
import { ImageType } from "../types";

// App Param Lists
export type AppParamsList = {
  Details: {
    prediction: Prediction;
    image: ImageType;
  };
  Landing: undefined;
  Home: undefined;
};

export type AppNavProps<T extends keyof AppParamsList> = {
  navigation: StackNavigationProp<AppParamsList, T>;
  route: RouteProp<AppParamsList, T>;
};
