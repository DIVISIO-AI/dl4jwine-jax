package divisio.dl4jwine;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.converters.FileConverter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.time.LocalDateTime;

/**
 * Performs testing of a trained & persisted model
 */
public class TestingApp {

    private static final Logger log = LoggerFactory.getLogger(TestingApp.class);

    @Parameter(names = {"-h", "--help"}, description = "Show usage info.", help = true)
    private boolean help;
    @Parameter(names = { "-pf", "--preprocessingFolder" }, description = "Folder for the preprocessed data.", converter = FileConverter.class)
    private File preprocessingFolder = new File("data/preprocessed");
    @Parameter(names = { "-of", "--outputFolder" }, description = "Folder with the trained model to test.", converter = FileConverter.class)
    private File outputFolder;
    @Parameter(names = { "-se", "--startEpoch" }, description = "Epoch to resume. -1: use last available.")
    private int startEpoch = -1;

    public static void main(final String[] args ) throws Exception {
        final TestingApp app = new TestingApp();

        // parse command line params
        final JCommander commander = JCommander.newBuilder().addObject(app).build();
        commander.parse(args);
        if (app.help) {
            commander.usage();
            System.exit(0);
            return;
        }

        final Trainer trainer = new MultilayerTrainer(app.preprocessingFolder);

        // check epoch state to restore (will default to null if there is none, causing a fresh training start)
        final File epochFile = TrainingApp.findEpochFile(app.outputFolder, app.startEpoch);
        int currentEpoch = TrainingApp.parseEpoch(epochFile);

        final LocalDateTime dateTime = LocalDateTime.now();
        final File logFile = new File(app.outputFolder,
                TrainingApp.DATE_TIME_FORMATTER.format(dateTime) + "_" + currentEpoch + "_test" + ".log");

        TrainingApp.initLogFile(logFile);
        log.info("Using log file " + logFile);

        trainer.test();
    }
}
