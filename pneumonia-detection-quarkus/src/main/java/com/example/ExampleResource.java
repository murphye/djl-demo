package com.example;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.Arrays;
import java.util.List;
import java.util.ServiceLoader;

import javax.ws.rs.GET;
import javax.ws.rs.Path;
import javax.ws.rs.Produces;
import javax.ws.rs.core.MediaType;

import ai.djl.MalformedModelException;
import ai.djl.engine.Engine;
import ai.djl.engine.EngineProvider;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.util.BufferedImageUtils;
import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.repository.zoo.ZooProvider;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;

@Path("/")
public class ExampleResource {

    @Path("/detect")
    @GET
    @Produces(MediaType.TEXT_PLAIN)
    public String detect() throws TranslateException, MalformedURLException, IOException, ModelNotFoundException,
            MalformedModelException {

        Criteria<BufferedImage, Classifications> criteria =
        Criteria.builder()
                .setTypes(BufferedImage.class, Classifications.class)
                .optEngine("TensorFlow")
                .optTranslator(new MyTranslator())
                .optProgress(new ProgressBar())
                .build();

        try (ZooModel<BufferedImage, Classifications> model = ModelZoo.loadModel(criteria)) {
            try (Predictor<BufferedImage, Classifications> predictor = model.newPredictor()) {

                var imagePath = "https://github.com/ieee8023/covid-chestxray-dataset/blob/master/images/01E392EE-69F9-4E33-BFCE-E5C968654078.jpeg?raw=true";
                var image = BufferedImageUtils.fromUrl(new URL(imagePath));

                Classifications result = predictor.predict(image);
                return result.toString();
            }
        }
    }

    @Path("/check")
    @GET
    @Produces(MediaType.TEXT_PLAIN)
    public String check() throws TranslateException, MalformedURLException, IOException, ModelNotFoundException,
            MalformedModelException {

        ServiceLoader<ZooProvider> providers = ServiceLoader.load(ZooProvider.class);

        for (ZooProvider provider : providers) {
            System.out.println(">>>>>>>> ZooProvider: " + provider.getName() + " " + provider.getModelZoo().getModelLoaders().toString());
        }
        
        ServiceLoader<EngineProvider> loaders = ServiceLoader.load(EngineProvider.class);
        
        for (EngineProvider provider : loaders) { 
            Engine engine = provider.getEngine(); 
            System.out.println(">>>>>>>> Engine: " + engine.getEngineName()); 
        }

        return "success";
    }

    private static final class MyTranslator implements Translator<BufferedImage, Classifications> {

        private static final List<String> CLASSES = Arrays.asList("Normal", "Pneumonia");

        @Override
        public NDList processInput(TranslatorContext ctx, BufferedImage input) {
            NDArray array =
                    BufferedImageUtils.toNDArray(
                            ctx.getNDManager(), input, NDImageUtils.Flag.COLOR);
            array = NDImageUtils.resize(array, 224).div(255.0f);
            return new NDList(array);
        }

        @Override
        public Classifications processOutput(TranslatorContext ctx, NDList list) {
            NDArray probabilities = list.singletonOrThrow();
            return new Classifications(CLASSES, probabilities);
        }
    }
}