# Quarkus + DJL Demo

Note: Tested with GraalVM 20.1.0 on Mac (https://www.graalvm.org/docs/reference-manual/native-image/)
Note: Error remains with Native executable (see below)

```
mkdir models
cd models
curl https://djl-tensorflow-javacpp.s3.amazonaws.com/tensorflow-models/chest_x_ray/saved_model.zip | jar xv
cd ..

mvn clean install -Dai.djl.repository.zoo.location=models/saved_model -Pnative
target/pneumonia-detection-quarkus-1.0.0-SNAPSHOT-runner -Dai.djl.repository.zoo.location=models/saved_model
```

`curl localhost:8080/detect` attempts to run the model (results in `ai.djl.engine.EngineException: No deep learning engine found.`)

`curl localhost:8080/check` demonstrates that the TensorFlow Engine is loaded via ServiceLoader mechanism. This contradicts the error from `/detect`

# pneumonia-detection-quarkus project

This project uses Quarkus, the Supersonic Subatomic Java Framework.

If you want to learn more about Quarkus, please visit its website: https://quarkus.io/ .

## Running the application in dev mode

You can run your application in dev mode that enables live coding using:
```
./mvnw quarkus:dev
```

## Packaging and running the application

The application can be packaged using `./mvnw package`.
It produces the `pneumonia-detection-quarkus-1.0.0-SNAPSHOT-runner.jar` file in the `/target` directory.
Be aware that it’s not an _über-jar_ as the dependencies are copied into the `target/lib` directory.

The application is now runnable using `java -jar target/pneumonia-detection-quarkus-1.0.0-SNAPSHOT-runner.jar`.

## Creating a native executable

You can create a native executable using: `./mvnw package -Pnative`.

Or, if you don't have GraalVM installed, you can run the native executable build in a container using: `./mvnw package -Pnative -Dquarkus.native.container-build=true`.

You can then execute your native executable with: `./target/pneumonia-detection-quarkus-1.0.0-SNAPSHOT-runner`

If you want to learn more about building native executables, please consult https://quarkus.io/guides/building-native-image.
