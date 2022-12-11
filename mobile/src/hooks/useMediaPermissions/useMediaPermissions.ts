import * as ImagePicker from "expo-image-picker";
import * as Camera from "expo-camera";
import { useEffect, useState } from "react";

const useMediaPermissions = () => {
  const [camera, setCamera] = useState(false);
  const [library, setLibrary] = useState(false);

  useEffect(() => {
    let mounted = true;
    if (mounted) {
      (async () => {
        const { granted } = await ImagePicker.getMediaLibraryPermissionsAsync();
        if (granted) {
          setLibrary(granted);
        } else {
          const { granted: g } =
            await ImagePicker.requestMediaLibraryPermissionsAsync();
          setLibrary(g);
        }
      })();
    }
    return () => {
      mounted = false;
    };
  }, []);
  useEffect(() => {
    let mounted = true;
    if (mounted) {
      (async () => {
        const { granted } = await Camera.requestCameraPermissionsAsync();
        setCamera(granted);
      })();
    }

    return () => {
      mounted = false;
    };
  }, []);
  return {
    camera,
    library,
  };
};

export default useMediaPermissions;
