package divisio.dl4jwine;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.api.storage.StatsStorageRouter;
import org.deeplearning4j.api.storage.impl.RemoteUIStatsStorageRouter;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.RegressionEvaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;

public class MultilayerTrainer implements Trainer {

    private static final Logger log = LoggerFactory.getLogger(MultilayerTrainer.class);

    private static final int PRINT_ITERATIONS = 500;

    private final File preprocessingFolder;

    //hyperparameters
    private final int nInputFeatures = 12;
    private final int outputFeatures = 1;
    private final int idxOutputFeature = nInputFeatures;//output feature is last column
    private final int[] layerWidths = new int[]{nInputFeatures, 128, 64, 32, 16, outputFeatures};
    private final WeightInit init = WeightInit.XAVIER;
    private final int batchSize = 256;

    private MultiLayerNetwork nn;
    private final StatsStorageRouter remoteUIRouter = new RemoteUIStatsStorageRouter("http://localhost:9000");

    public MultilayerTrainer(final File preprocessingFolder) throws Exception {
        this.preprocessingFolder = preprocessingFolder;

        //build configuration
        final MultiLayerConfiguration nnConf = new NeuralNetConfiguration.Builder()
                .seed(12345678)
                .weightInit(init)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new RmsProp())
                .dropOut(0.3)
                .l2(0.1)
                // ... other hyperparameters
                .list(
                    buildLayers(layerWidths, Activation.RELU, LossFunctions.LossFunction.L2)
                )
                .pretrain(false)
                //.inputPreProcessor() // if necessary, add additional preprocessing here
                .backprop(true).build();

        //build network
        nn = new MultiLayerNetwork(nnConf);

        attachListeners();
    }

    private Layer[] buildLayers(final int[] layerWidths, final Activation activation, final LossFunction loss) {
        final Layer[] result = new Layer[layerWidths.length - 1];
        //hidden layers
        for (int i = 0; i < layerWidths.length - 2; ++i) {
            result[i] = new DenseLayer.Builder()
                    .nIn(layerWidths[i]).nOut(layerWidths[i + 1])
                    .activation(activation).build();
        }
        //output layer
        result[result.length - 1] = new OutputLayer.Builder(loss)
                .nIn(layerWidths[layerWidths.length - 2]).nOut(layerWidths[layerWidths.length - 1])
                .activation(Activation.IDENTITY).build();

        return result;
    }

    private void attachListeners() {
        nn.setListeners(new StatsListener(remoteUIRouter), //sends stats to the UI
                        new ScoreIterationListener(PRINT_ITERATIONS));//logs scores
    }

    @Override
    public String getTag() {
        return "multilayer_less_overfit_large_batch_size";
    }

    private DataSetIterator buildIterator(final File csvFile) throws IOException, InterruptedException {
        final RecordReader rr = new CSVRecordReader();
        rr.initialize(new FileSplit(csvFile));
        return new RecordReaderDataSetIterator(
                rr, null, batchSize,
                idxOutputFeature, idxOutputFeature, -1, -1,
                true);
    }

    @Override
    public void train() throws Exception {
        final File trainingFile   = new File(preprocessingFolder, "training.csv");
        final DataSetIterator iter = buildIterator(trainingFile);
        nn.fit(iter);
    }

    private void runEvaluation(final File file) throws Exception {
        final DataSetIterator iter = buildIterator(file);
        final RegressionEvaluation evaluation = nn.evaluateRegression(iter);
        log.info("\n" + evaluation.stats());
        // TODO: it can make sense to run each example manually here, print individual results so we can identify which
        // cases are especially problematic
        // TODO: create error histogram
        // TODO: in our case, an error per rating value makes sense, as our regression target has a lot of values of 5/6/7
        // and few extremes (3,4,8,9)
        //
    }

    @Override
    public void validate() throws Exception {
        runEvaluation(new File(preprocessingFolder, "validation.csv"));
    }

    @Override
    public void test() throws Exception {
        final File testFile = new File(preprocessingFolder, "testing.csv");
        runEvaluation(testFile);
    }

    @Override
    public void loadState(final File file) throws IOException {
        nn = ModelSerializer.restoreMultiLayerNetwork(file, true);
        attachListeners();
    }

    @Override
    public void saveState(final File file) throws IOException {
        if (nn != null) {
            ModelSerializer.writeModel(nn, file, true);
        }
    }
}
