package com.example;

import ai.djl.Application;
import ai.djl.ModelException;
import ai.djl.engine.Engine;
import ai.djl.engine.EngineProvider;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.translator.ImageClassificationTranslator;
import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.repository.Artifact;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.repository.zoo.ZooProvider;
import ai.djl.tensorflow.engine.LibUtils;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.tensorflow.EagerSession;

import javax.ws.rs.GET;
import javax.ws.rs.Path;
import javax.ws.rs.Produces;
import javax.ws.rs.core.MediaType;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.ServiceLoader;

@Path("/")
public class ExampleResource {

    private static final Logger logger = LoggerFactory.getLogger(ExampleResource.class);

    private static final List<String> CLASSES = Arrays.asList("Normal", "Pneumonia");

    @Path("/detect")
    @GET
    @Produces(MediaType.TEXT_PLAIN)
    public String detect() throws TranslateException, IOException, ModelException {
        LibUtils.loadLibrary();
        EagerSession.getDefault();

        Translator<Image, Classifications> translator = ImageClassificationTranslator.builder()
                .addTransform(a -> NDImageUtils.resize(a, 224).div(255.0f))
                .optSynset(CLASSES)
                .build();
        Criteria<Image, Classifications> criteria =
                Criteria.builder()
                        .setTypes(Image.class, Classifications.class)
                        .optModelName("saved_model")
                        .optEngine("TensorFlow")
                        .optTranslator(translator)
                        .build();

        try (ZooModel<Image, Classifications> model = ModelZoo.loadModel(criteria);
            Predictor<Image, Classifications> predictor = model.newPredictor()) {
            var imagePath = "https://djl-ai.s3.amazonaws.com/resources/images/chest_xray.jpg";
            var image = ImageFactory.getInstance().fromUrl(imagePath);

            Classifications result = predictor.predict(image);
            return result.toString();
        }
    }

    @Path("/check")
    @GET
    @Produces(MediaType.TEXT_PLAIN)
    public String check() throws IOException, ModelNotFoundException {
        ServiceLoader<ZooProvider> providers = ServiceLoader.load(ZooProvider.class);

        Map<Application, List<Artifact>> models = ModelZoo.listModels();
        models.forEach(
                (app, list) -> {
                    String appName = app.getPath().replace('/', '.').toUpperCase();
                    list.forEach(artifact -> logger.info("{} {}", appName, artifact));
                });

        ServiceLoader<EngineProvider> loaders = ServiceLoader.load(EngineProvider.class);

        for (EngineProvider provider : loaders) {
            Engine engine = provider.getEngine();
            System.out.println(">>>>>>>> Engine: " + engine.getEngineName());
        }

        return "success";
    }
}
