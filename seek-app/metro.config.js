const { getDefaultConfig } = require('expo/metro-config');

const config = getDefaultConfig(__dirname);
// ponytail: bundle .onnx model as a static asset; swap to runtime download if APK size becomes a problem
config.resolver.assetExts.push('onnx', 'ort');

module.exports = config;
