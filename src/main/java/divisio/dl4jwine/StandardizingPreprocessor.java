package divisio.dl4jwine;

import com.google.common.base.Charsets;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.records.writer.impl.csv.CSVRecordWriter;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.partition.NumberOfRecordsPartitioner;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.analysis.DataAnalysis;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.transform.normalize.Normalize;
import org.datavec.api.transform.ui.HtmlAnalysis;
import org.datavec.api.writable.Writable;
import org.datavec.local.transforms.LocalTransformExecutor;
import org.datavec.spark.transform.AnalyzeSpark;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;


public class StandardizingPreprocessor implements Preprocessor {

    private static final Logger log = LoggerFactory.getLogger(StandardizingPreprocessor.class);

    public static final Schema inputSchema = new Schema.Builder()
            // Input variables (based on physicochemical tests):
            // 1 - fixed acidity
            .addColumnDouble("fixed acidity")
            // 2 - volatile acidity
            .addColumnDouble("volatile acidity")
            // 3 - citric acid
            .addColumnDouble("citric acid")
            // 4 - residual sugar
            .addColumnDouble("residual sugar")
            // 5 - chlorides
            .addColumnDouble("chlorides")
            // 6 - free sulfur dioxide
            .addColumnDouble("free sulfur dioxide")
            // 7 - total sulfur dioxide
            .addColumnDouble("total sulfur dioxide")
            // 8 - density
            .addColumnDouble("density")
            // 9 - pH
            .addColumnDouble("pH")
            // 10 - sulphates
            .addColumnDouble("sulphates")
            // 11 - alcohol
            .addColumnDouble("alcohol")
            // Output variable (based on sensory data):
            // 12 - quality (score between 0 and 10)
            .addColumnInteger("quality")
            .build();
    //NOTE: do *not* use addColumnFloat - the column type cannot be analyzed and will cause an "Unknown column type: Float"
    // error

    private final File rawDataFolder;
    private final File preprocessingFolder;
    private DataAnalysis dataAnalysisRaw;
    private DataAnalysis dataAnalysisStandardized;

    public StandardizingPreprocessor(final File rawDataFolder, final File preprocessingFolder) {
        this.rawDataFolder = rawDataFolder;
        this.preprocessingFolder = preprocessingFolder;
    }

    @Override
    public String getTag() {
        // a short-ish memorable description of what this class does
        return "add_wine_type_shuffle_standardize";
    }

    @Override
    public void writeLog(final File logFolder) throws Exception {
        // write info about preprocessing to model dir for reference
        if (dataAnalysisRaw != null) {
            HtmlAnalysis.createHtmlAnalysisFile(dataAnalysisRaw, new File(logFolder, "DataAnalysisRaw.html"));
        }
        if (dataAnalysisStandardized != null) {
            HtmlAnalysis.createHtmlAnalysisFile(dataAnalysisStandardized, new File(logFolder, "DataAnalysisStandardized.html"));
        }
    }

    private List<List<Writable>> readAll(final RecordReader rr) {
        final ArrayList<List<Writable>> result = new ArrayList<>();
        while (rr.hasNext()) {
            result.add(rr.next());
        }
        return result;
    }

    private void writeAll(final List<List<Writable>> records, final File file) throws Exception {
        file.getParentFile().mkdirs();
        final Writer out = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(file), Charsets.UTF_8));
        try {
            final Iterator<List<Writable>> iRecords = records.iterator();
            while (iRecords.hasNext()) {
                final Iterator<Writable> iRecord = iRecords.next().iterator();
                while (iRecord.hasNext()) {
                    out.write(iRecord.next().toString());
                    if (iRecord.hasNext()) {
                        out.write(',');
                    }
                }
                if (iRecords.hasNext()) {
                    out.write('\n');
                }
            }
        } catch (final Exception e) {
            file.delete();
            throw e;
        } finally {
            out.close();
        }

    }

    @Override
    public void preprocess() throws Exception {

        final File trainingFile   = new File(preprocessingFolder, "training.csv");
        final File validationFile = new File(preprocessingFolder, "validation.csv");
        final File testingFile    = new File(preprocessingFolder, "testing.csv");

        // check if target files all exist
        if (trainingFile.isFile() && validationFile.isFile() && testingFile.isFile()) {
            log.info("Output files exist, skipping preprocessing.");
            return;
        }
        log.info("Output does not exist, starting preprocessing.");

        // if not, read input CSVs
        final File whiteWineFile = new File(rawDataFolder, "winequality-red.csv");
        final File redWineFile   = new File(rawDataFolder, "winequality-white.csv");

        //Load the data:
        CSVRecordReader rr = new CSVRecordReader(1, ';', '"');
        rr.initialize(new FileSplit(whiteWineFile));
        final List<List<Writable>> whiteWine = readAll(rr);
        // we have to re-create the reader, as otherwise the header is not skipped -.-
        rr = new CSVRecordReader(1, ';', '"');
        rr.initialize(new FileSplit(redWineFile));
        final List<List<Writable>> redWine = readAll(rr);

        // transform to add wine type
        final TransformProcess tpWhite = new TransformProcess.Builder(inputSchema)
                .addConstantIntegerColumn("wine type", 0)
                .build();
        final TransformProcess tpRed = new TransformProcess.Builder(inputSchema)
                .addConstantIntegerColumn("wine type", 1)
                .build();

        //process & concatenate
        final List<List<Writable>> allWineWithType = new ArrayList<>();
        allWineWithType.addAll(LocalTransformExecutor.execute(whiteWine, tpWhite));
        allWineWithType.addAll(LocalTransformExecutor.execute(redWine,   tpRed));

        // shuffle
        Collections.shuffle(allWineWithType);

        // split into testing, validation & training data
        // TODO: add stratified sampling example so wine type & result scores are properly distributed in each set
        // TODO: remove outliers, add clamping to min/max values
        final int splitpoint1 = allWineWithType.size() / 10;
        final int splitpoint2 = splitpoint1 * 2;
        List<List<Writable>> testing    = allWineWithType.subList(0, splitpoint1);
        List<List<Writable>> validation = allWineWithType.subList(splitpoint1, splitpoint2);
        List<List<Writable>> training   = allWineWithType.subList(splitpoint2, allWineWithType.size());

        // determine normalization parameters from training data
        SparkConf conf = new SparkConf();
        conf.setMaster("local[*]");
        conf.setAppName(getTag());
        JavaSparkContext sc = new JavaSparkContext(conf);

        JavaRDD<List<Writable>> trainingRdd = sc.parallelize(training);
        int maxHistogramBuckets = 50;
        dataAnalysisRaw = AnalyzeSpark.analyze(tpRed.getFinalSchema(), trainingRdd, maxHistogramBuckets);
        log.info(dataAnalysisRaw.toString());

        // apply normalization to all necessary columns
        final TransformProcess tpNormalize = new TransformProcess.Builder(tpRed.getFinalSchema())
                .reorderColumns("wine type")//move wine type to front, so quality is last again
                .normalize("fixed acidity", Normalize.Standardize, dataAnalysisRaw)
                .normalize("volatile acidity", Normalize.Standardize, dataAnalysisRaw)
                .normalize("citric acid", Normalize.Standardize, dataAnalysisRaw)
                .normalize("residual sugar", Normalize.Standardize, dataAnalysisRaw)
                .normalize("chlorides", Normalize.Standardize, dataAnalysisRaw)
                .normalize("free sulfur dioxide", Normalize.Standardize, dataAnalysisRaw)
                .normalize("total sulfur dioxide", Normalize.Standardize, dataAnalysisRaw)
                .normalize("density", Normalize.Standardize, dataAnalysisRaw)
                .normalize("pH", Normalize.Standardize, dataAnalysisRaw)
                .normalize("sulphates", Normalize.Standardize, dataAnalysisRaw)
                .normalize("alcohol", Normalize.Standardize, dataAnalysisRaw)
                .build();

        testing    = LocalTransformExecutor.execute(testing, tpNormalize);
        validation = LocalTransformExecutor.execute(validation, tpNormalize);
        training   = LocalTransformExecutor.execute(training, tpNormalize);

        trainingRdd = sc.parallelize(training);
        dataAnalysisStandardized = AnalyzeSpark.analyze(tpNormalize.getFinalSchema(), trainingRdd);
        log.info(dataAnalysisStandardized.toString());

        // write data to new CSVs
        writeAll(training, trainingFile);
        writeAll(validation, validationFile);
        writeAll(testing, testingFile);
    }
}
