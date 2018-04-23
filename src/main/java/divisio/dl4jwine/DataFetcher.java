package divisio.dl4jwine;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.net.MalformedURLException;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;

/**
 * Downloader for our raw data. Often your data will just be available on the file system or via a jdbc connection etc.,
 * so you do not need a class like this.
 */
public class DataFetcher {

    private static final Logger log = LoggerFactory.getLogger(DataFetcher.class);

    private final File rawDataFolder;

    /**
     * @param rawDataFolder target folder for downloaded data
     */
    public DataFetcher(final File rawDataFolder) {
        this.rawDataFolder = rawDataFolder;
    }

    /**
     * URL to our wine training data
     */
    private static final String baseUrl = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/";

    private File dataFile(final String name) { return new File(rawDataFolder, name); }

    private URL dataUrl(final String name) throws MalformedURLException { return new URL(baseUrl + name); }

    private void fetchFile(final String name) throws IOException {
        final File file = dataFile(name);
        // if we already downloaded the file we are done
        if (file.isFile()) {
            log.info(file + " exists, skipping download.");
            return;
        }
        // make sure the parent folder exists
        file.getParentFile().mkdirs();
        // download to file
        final URL url = dataUrl(name);
        log.info("Downloading from " + url + " to " + file);
        try (final InputStream in = url.openConnection().getInputStream()) {
            Files.copy(in, file.toPath(), StandardCopyOption.REPLACE_EXISTING);
        } catch (final Exception e) {
            // if something goes wrong, delete target file so the download will be attempted again on the next try
            file.delete();
            log.error("Error fetching file " + file + " from " + url, e);
            throw e;
        }
    }

    /**
     * fetches the wine quality data if necessary
     * @throws IOException
     */
    public void fetchData() throws IOException {
        fetchFile("winequality.names");
        fetchFile("winequality-red.csv");
        fetchFile("winequality-white.csv");
    }
}
