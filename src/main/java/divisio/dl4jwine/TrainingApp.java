package divisio.dl4jwine;

import ch.qos.logback.classic.Level;
import ch.qos.logback.classic.LoggerContext;
import ch.qos.logback.classic.encoder.PatternLayoutEncoder;
import ch.qos.logback.core.FileAppender;
import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.converters.FileConverter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

/**
 * Contains the data download, preprocessing & training steps
 */
public class TrainingApp {

    private static final Logger log = LoggerFactory.getLogger(TrainingApp.class);

    private static final String EPOCH_FILE_PREFIX = "epoch_";
    private static final String EPOCH_FILE_SUFFIX = ".zip";
    public static final DateTimeFormatter DATE_TIME_FORMATTER = DateTimeFormatter.ofPattern("yyyy-MM-dd HH-mm-ss");

    @Parameter(names = {"-h", "--help"}, description = "Show usage info.", help = true)
    private boolean help;
    @Parameter(names = { "-rf", "--rawDataFolder" }, description = "Folder with the raw data.", converter = FileConverter.class)
    private File rawDataFolder = new File("data/raw");
    @Parameter(names = { "-pf", "--preprocessingFolder" }, description = "Folder for the preprocessed data.", converter = FileConverter.class)
    private File preprocessingFolder = new File("data/preprocessed");
    @Parameter(names = { "-mf", "--modelFolder" }, description = "Folder for the trained models.", converter = FileConverter.class)
    private File modelFolder = new File("data/model");
    @Parameter(names = { "-se", "--startEpoch" }, description = "Epoch to resume. -1: use last available.")
    private int startEpoch = -1;
    @Parameter(names = {"-e", "--epochs"}, description = "Number of epochs to train")
    private int epochs = 1000;

    public static File buildOutputFolder(final File modelFolder, final Preprocessor preprocessor, final Trainer trainer) {
        return new File(modelFolder, preprocessor.getTag() + "/" + trainer.getTag());
    }

    public static File buildEpochFile(final File outputFolder, final int epoch) {
        return new File(outputFolder, EPOCH_FILE_PREFIX + epoch + EPOCH_FILE_SUFFIX);
    }

    public static File findLastEpochFile(final File outputFolder) {
        int maxEpoch = Integer.MIN_VALUE;
        File maxEpochFile = null;
        final File[] files = outputFolder.listFiles();
        if (files == null) { return null; }
        for (final File file : files) {
            if (file.getName().startsWith(EPOCH_FILE_PREFIX) && file.getName().endsWith(EPOCH_FILE_SUFFIX)) {
                final int epoch = parseEpoch(file);
                if (epoch > maxEpoch) {
                    maxEpoch = epoch;
                    maxEpochFile = file;
                }
            }
        }
        return maxEpochFile;
    }

    public static File findEpochFile(final File outputFolder, final int startEpoch) {
        if (startEpoch == -1) {
            // find last epoch state
            return findLastEpochFile(outputFolder);
        } else {
            // resume particular epoch state
            return buildEpochFile(outputFolder, startEpoch);
        }
    }

    public static int parseEpoch(final File epochFile) {
        if (epochFile == null) { return 0; }
        final String name = epochFile.getName();
        return Integer.parseInt(name.substring(name.indexOf('_') + 1, name.indexOf('.')));
    }

    public static void initLogFile(final File logFile) {
        final LoggerContext loggerContext = (LoggerContext) LoggerFactory.getILoggerFactory();

        final FileAppender fileAppender = new FileAppender();
        fileAppender.setContext(loggerContext);
        fileAppender.setName("file");
        fileAppender.setFile(logFile.getAbsolutePath());

        final PatternLayoutEncoder encoder = new PatternLayoutEncoder();
        encoder.setContext(loggerContext);
        encoder.setPattern("%date{ISO8601} %logger{24} %level - %msg%n");
        encoder.start();

        fileAppender.setEncoder(encoder);
        fileAppender.start();

        //append new logger as additional appender for root logging
        final ch.qos.logback.classic.Logger rootLogger =
                loggerContext.getLogger(ch.qos.logback.classic.Logger.ROOT_LOGGER_NAME);
        rootLogger.setLevel(Level.INFO);
        rootLogger.addAppender(fileAppender);
    }

    public static void main(final String[] args ) throws Exception {
        // create instance of our AI Application
        final TrainingApp app = new TrainingApp();

        // parse command line params
        final JCommander commander = JCommander.newBuilder().addObject(app).build();
        commander.parse(args);
        if (app.help) {
            commander.usage();
            System.exit(0);
            return;
        }

        // create trainer and preprocessor
        final Preprocessor preprocessor = new StandardizingPreprocessor(app.rawDataFolder, app.preprocessingFolder);
        final Trainer trainer = new MultilayerTrainer(app.preprocessingFolder);

        // determine model folder so we know where to write log info and models to
        // (this is a subfolder of the given model folder, one subfolder for each preprocessor / trainer combination)
        // modelFolder/[preprocessing_version]/[model_version]/
        // {schema file, transformation, trained model files, training log with timestamp}
        final File outputFolder = buildOutputFolder(app.modelFolder, preprocessor, trainer);
        outputFolder.mkdirs();

        // check epoch state to restore (will default to null if there is none, causing a fresh training start)
        final File epochFile = findEpochFile(outputFolder, app.startEpoch);
        int currentEpoch = parseEpoch(epochFile);

        // determine log file name from epoch name and start time
        final LocalDateTime dateTime = LocalDateTime.now();
        final File logFile = new File(outputFolder,
                DATE_TIME_FORMATTER.format(dateTime) + "_" + currentEpoch + "_" + "train" + ".log");

        // adjust logger so we also write log to model folder as well as to stderr
        initLogFile(logFile);
        log.info("Writing output to " + outputFolder);
        log.info("Using log file " + logFile);

        // make sure our data is available
        new DataFetcher(app.rawDataFolder).fetchData();

        // preprocess data so we can feed it to our network
        preprocessor.preprocess();
        preprocessor.writeLog(outputFolder);

        // restore last epoch, if there is one
        if (epochFile != null) {
            log.info("Resuming from epoch file: " + epochFile);
            if (!(epochFile.isFile() && epochFile.canRead())) {
                log.error("Cannot read epoch file " + epochFile);
                System.exit(-1);
                return;
            } else {
                trainer.loadState(epochFile);
            }
        } else {
            log.info("No epoch to resume, starting from scratch.");
        }

        // run training
        final int epochSaveStep = 100; // after how many epochs we want to save (1 == save every epoch)
        for (int i = 0; i < app.epochs; ++i) {
            currentEpoch++;
            final long start = System.currentTimeMillis();
            trainer.train();
            log.info("Epoch " + currentEpoch + " took " + ((System.currentTimeMillis() - start)) + "ms to train.");
            if ((i + 1) % epochSaveStep == 0 && i > 0) {
                trainer.saveState(buildEpochFile(outputFolder, currentEpoch));
                trainer.validate();
            }
        }


        // TODO github version:
        // - questions: What about shuffling each epoch?
        // -

        // TODO internal version:
        // - switch maven -> gradle
        // - switch java -> kotlin
        // - add scripts for deploying dependencies, add scripts for deploying artifact, add script for training & resuming
        // - add defaults for parameters so they fit directories on ai machine
        // - add custom preprocessing (centered log, clamp)
        // - check shuffling, maybe implement custom CSV reader
        // - create LocalAnalyzer to create analysis without spark
    }
}
